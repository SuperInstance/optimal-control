//! Linear Quadratic Regulator — the workhorse of optimal control.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use crate::riccati::solve_dare;
use crate::dynamics::LinearSystem;

/// LQR cost weights: minimize sum of x'Qx + u'Ru
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LqrCost {
    pub q: DMatrix<f64>,  // state weight (n x n), positive semi-definite
    pub r: DMatrix<f64>,  // control weight (m x m), positive definite
    pub n_matrix: Option<DMatrix<f64>>,  // cross term (n x m), optional
}

impl LqrCost {
    pub fn new(q: DMatrix<f64>, r: DMatrix<f64>) -> Self {
        Self { q, r, n_matrix: None }
    }

    pub fn with_cross_term(mut self, n: DMatrix<f64>) -> Self {
        self.n_matrix = Some(n);
        self
    }
}

/// LQR solution: optimal gain K and associated Riccati solution P
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LqrSolution {
    pub k: DMatrix<f64>,  // optimal gain (m x n)
    pub p: DMatrix<f64>,  // Riccati solution (n x n)
}

impl LqrSolution {
    /// Compute optimal control: u = -K*x
    pub fn control(&self, x: &DVector<f64>) -> DVector<f64> {
        -&self.k * x
    }

    /// Compute optimal cost-to-go from state x: V(x) = x'Px
    pub fn cost_to_go(&self, x: &DVector<f64>) -> f64 {
        let px = &self.p * x;
        x.dot(&px)
    }
}

/// Solve the continuous-time LQR problem for system (A, B) with cost (Q, R).
/// Uses the algebraic Riccati equation.
pub fn solve_lqr(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> Result<LqrSolution, String> {
    let n = a.nrows();
    let m = b.ncols();

    // Build Hamiltonian matrix and solve
    // [A  -B R^{-1} B'] [P]
    // [-Q   -A'        ] [I] = 0
    let r_inv = r.clone().try_inverse().ok_or("R is not invertible")?;
    let brinvbt = b * &r_inv * b.transpose();

    // Use iterative Riccati solution
    let p = solve_care(a, b, q, r)?;

    // K = R^{-1} B' P
    let k = &r_inv * b.transpose() * &p;

    Ok(LqrSolution { k, p })
}

/// Solve continuous algebraic Riccati equation.
/// Uses the Hamiltonian matrix approach with eigenvalue decomposition.
pub fn solve_care(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> Result<DMatrix<f64>, String> {
    let r_inv = r.clone().try_inverse().ok_or("R is not invertible")?;
    let brinvbt = b * &r_inv * b.transpose();
    let n = a.nrows();

    // Use a well-scaled initial guess
    // Scale P so that A-BK is stable where K = R^{-1}B'P
    let mut p = q.clone() * 10.0; // Start with larger P to ensure ak is stable

    // Apply Newton's method for CARE
    for _ in 0..5000 {
        let residual = a.transpose() * &p + &p * a - &p * &brinvbt * &p + q;
        if residual.norm() < 1e-10 {
            return Ok(p);
        }

        let ak = a - &brinvbt * &p;
        let i_n = DMatrix::identity(n, n);

        let lhs = kronecker(&i_n, &ak.transpose()) + kronecker(&ak.transpose(), &i_n);
        let rhs = -vectorize(&residual);

        if let Some(dp_vec) = lhs.lu().solve(&rhs) {
            let dp = unvectorize(&dp_vec, n, n);
            if dp.norm() < p.norm() * 10.0 + 1.0 {
                p = p + dp;
                continue;
            }
        }

        // Gradient descent fallback
        let step = 0.0001 / (1.0 + residual.norm());
        p = p - step * &residual;
    }

    let residual = a.transpose() * &p + &p * a - &p * &brinvbt * &p + q;
    if residual.norm() < 1e-6 {
        Ok(p)
    } else {
        Err(format!("CARE did not converge, residual norm: {}", residual.norm()))
    }
}

/// Solve discrete-time LQR for system (A_d, B_d) with cost (Q, R).
/// Minimizes sum_k x_k' Q x_k + u_k' R u_k
pub fn solve_dlqr(
    ad: &DMatrix<f64>,
    bd: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> Result<LqrSolution, String> {
    let p = solve_dare(ad, bd, q, r)?;
    let r_inv = r.clone().try_inverse().ok_or("R is not invertible")?;
    let k = (r.clone() + bd.transpose() * &p * bd)
        .try_inverse()
        .ok_or("R + B'PB not invertible")?
        * bd.transpose()
        * &p
        * ad;

    Ok(LqrSolution { k, p })
}

/// Compute LQR trajectory for a linear system
pub fn lqr_trajectory(
    sys: &LinearSystem,
    cost: &LqrCost,
    x0: &DVector<f64>,
    dt: f64,
    n_steps: usize,
) -> Result<(Vec<DVector<f64>>, Vec<DVector<f64>>), String> {
    // Discretize: A_d ≈ I + dt*A, B_d ≈ dt*B
    let n = sys.state_dim();
    let m = sys.control_dim();
    let ad = DMatrix::identity(n, n) + dt * &sys.a;
    let bd = dt * &sys.b;

    let sol = solve_dlqr(&ad, &bd, &cost.q, &cost.r)?;

    let mut states = Vec::with_capacity(n_steps + 1);
    let mut controls = Vec::with_capacity(n_steps);
    states.push(x0.clone());
    let mut x = x0.clone();

    for _ in 0..n_steps {
        let u = sol.control(&x);
        x = &ad * &x + &bd * &u;
        states.push(x.clone());
        controls.push(u);
    }

    Ok((states, controls))
}

/// Kronecker product of two matrices
pub fn kronecker(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let (ra, ca) = a.shape();
    let (rb, cb) = b.shape();
    let mut result = DMatrix::zeros(ra * rb, ca * cb);
    for i in 0..ra {
        for j in 0..ca {
            for p in 0..rb {
                for q in 0..cb {
                    result[(i * rb + p, j * cb + q)] = a[(i, j)] * b[(p, q)];
                }
            }
        }
    }
    result
}

/// Vectorize a matrix column-major
pub fn vectorize(m: &DMatrix<f64>) -> DVector<f64> {
    let (r, c) = m.shape();
    let mut v = Vec::with_capacity(r * c);
    for j in 0..c {
        for i in 0..r {
            v.push(m[(i, j)]);
        }
    }
    DVector::from_vec(v)
}

/// Un-vectorize into (r, c) matrix
pub fn unvectorize(v: &DVector<f64>, r: usize, c: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(r, c);
    let mut idx = 0;
    for j in 0..c {
        for i in 0..r {
            m[(i, j)] = v[idx];
            idx += 1;
        }
    }
    m
}

/// Compute total quadratic cost along a trajectory
pub fn trajectory_cost(
    states: &[DVector<f64>],
    controls: &[DVector<f64>],
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> f64 {
    let mut cost = 0.0;
    for (i, x) in states.iter().enumerate() {
        let qx = q * x;
        cost += x.dot(&qx);
        if i < controls.len() {
            let u = &controls[i];
            let ru = r * u;
            cost += u.dot(&ru);
        }
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_lqr_gain_stabilizes() {
        // Use discrete system directly
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);

        let sol = solve_dlqr(&a, &b, &q, &r).unwrap();
        let a_cl = &a - &b * &sol.k;
        let eig = a_cl.complex_eigenvalues();
        for i in 0..eig.len() {
            let mag = (eig[i].re * eig[i].re + eig[i].im * eig[i].im).sqrt();
            assert!(mag < 1.0, "Eigenvalue magnitude should be < 1: {}", mag);
        }
    }

    #[test]
    fn test_lqr_cost_to_go_positive() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);

        let sol = solve_dlqr(&a, &b, &q, &r).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let v = sol.cost_to_go(&x);
        assert!(v > 0.0, "Cost-to-go should be positive");
    }

    #[test]
    fn test_trajectory_cost_lqr() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);

        let sol = solve_dlqr(&a, &b, &q, &r).unwrap();
        let u = sol.control(&DVector::from_vec(vec![1.0, 0.0]));
        assert!(u[0] < 0.0, "Control should push toward origin");
    }

    #[test]
    fn test_kronecker_identity() {
        let a = DMatrix::identity(2, 2);
        let k = kronecker(&a, &a);
        assert_eq!(k, DMatrix::identity(4, 4));
    }

    #[test]
    fn test_vectorize_roundtrip() {
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = vectorize(&m);
        let m2 = unvectorize(&v, 2, 3);
        assert!((m - m2).norm() < 1e-10);
    }
}
