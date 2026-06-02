//! Riccati equation solvers (algebraic and differential).

use nalgebra::DMatrix;

/// Solve discrete-time algebraic Riccati equation iteratively.
/// P = A'PA - A'PB(R + B'PB)^{-1} B'PA + Q
pub fn solve_dare(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> Result<DMatrix<f64>, String> {
    let n = a.nrows();

    let mut p = q.clone();

    for _ in 0..1000 {
        let s = r + b.transpose() * &p * b;
        let s_inv = s.try_inverse().ok_or("R + B'PB not invertible")?;

        let p_new = a.transpose() * &p * a
            - a.transpose() * &p * b * &s_inv * b.transpose() * &p * a
            + q;

        let diff = (&p_new - &p).norm();
        p = p_new;

        if diff < 1e-12 {
            return Ok(p);
        }
    }

    // Verify convergence
    let s = r + b.transpose() * &p * b;
    let s_inv = match s.try_inverse() {
        Some(inv) => inv,
        None => return Err("DARE did not converge".to_string()),
    };
    let residual = a.transpose() * &p * a
        - a.transpose() * &p * b * &s_inv * b.transpose() * &p * a
        + q
        - &p;
    if residual.norm() < 1e-6 {
        Ok(p)
    } else {
        Err("DARE did not converge".to_string())
    }
}

/// Solve continuous-time algebraic Riccati equation via iterative method.
pub fn solve_care_iterative(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> Result<DMatrix<f64>, String> {
    crate::lqr::solve_care(a, b, q, r)
}

/// Solve the differential Riccati equation (finite-horizon).
/// Integrate backwards from T to 0 using Euler method.
pub fn solve_dre(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    qf: &DMatrix<f64>,
    t_final: f64,
    dt: f64,
) -> Vec<(f64, DMatrix<f64>)> {
    let r_inv = r.clone().try_inverse().expect("R not invertible");
    let brinvbt = b * &r_inv * b.transpose();

    let n_steps = (t_final / dt).ceil() as usize;
    let mut result = Vec::with_capacity(n_steps + 1);

    let mut p = qf.clone();
    let mut t = t_final;
    result.push((t, p.clone()));

    for _ in 0..n_steps {
        let dp = a.transpose() * &p + &p * a - &p * &brinvbt * &p + q;
        p = p + dt * dp;
        t -= dt;
        result.push((t, p.clone()));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_dare_simple_1d() {
        let a = DMatrix::from_row_slice(1, 1, &[1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);

        let p = solve_dare(&a, &b, &q, &r).unwrap();
        // P satisfies P^2 - P - 1 = 0 => P = (1+sqrt(5))/2 ≈ 1.618
        let p_analytical = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((p[(0, 0)] - p_analytical).abs() < 1e-4, "P = {}, expected {}", p[(0,0)], p_analytical);
    }

    #[test]
    fn test_dare_identity() {
        let a = DMatrix::identity(2, 2);
        let b = DMatrix::identity(2, 2);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::identity(2, 2);

        let p = solve_dare(&a, &b, &q, &r).unwrap();
        assert!((p.clone() - p.transpose()).norm() < 1e-8);
        let eig = p.symmetric_eigenvalues();
        for i in 0..eig.nrows() {
            assert!(eig[i] > 0.0);
        }
    }

    #[test]
    fn test_dare_double_integrator() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
        let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);

        let p = solve_dare(&a, &b, &q, &r).unwrap();
        assert!((p.clone() - p.transpose()).norm() < 1e-8);
        let eig = p.symmetric_eigenvalues();
        for i in 0..eig.nrows() {
            assert!(eig[i] > 0.0);
        }
    }

    #[test]
    fn test_care_identity() {
        let a = DMatrix::identity(2, 2) * 0.0;
        let b = DMatrix::identity(2, 2);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::identity(2, 2);

        let p = solve_care_iterative(&a, &b, &q, &r).unwrap();
        assert!((p - DMatrix::identity(2, 2)).norm() < 1e-6);
    }

    #[test]
    fn test_dre_converges_to_care() {
        let a = DMatrix::from_row_slice(2, 2, &[-0.5, 0.0, 0.0, -0.5]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let qf = DMatrix::identity(2, 2);

        let dre = solve_dre(&a, &b, &q, &r, &qf, 50.0, 0.01);
        let p_t0 = &dre.last().unwrap().1;
        let p_care = solve_care_iterative(&a, &b, &q, &r).unwrap();
        assert!((p_t0 - &p_care).norm() < 0.5);
    }

    #[test]
    fn test_dre_symmetry_preserved() {
        let a = DMatrix::from_row_slice(1, 1, &[0.5]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let qf = DMatrix::from_row_slice(1, 1, &[10.0]);

        let dre = solve_dre(&a, &b, &q, &r, &qf, 10.0, 0.01);
        for (_, p) in &dre {
            assert!((p.clone() - p.transpose()).norm() < 1e-8);
        }
    }

    #[test]
    fn test_dare_stabilizing() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);

        let p = solve_dare(&a, &b, &q, &r).unwrap();
        let k = b.transpose() * &p;

        let a_cl = &a - &b * &k;
        let eig = a_cl.complex_eigenvalues();
        for i in 0..eig.len() {
            let mag = (eig[i].re * eig[i].re + eig[i].im * eig[i].im).sqrt();
            assert!(mag < 1.0, "Eigenvalue magnitude {} should be < 1", mag);
        }
    }
}
