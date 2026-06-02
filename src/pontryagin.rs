//! Pontryagin's Maximum Principle: Hamiltonian maximization for optimal control.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Result of applying Pontryagin's Maximum Principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmpResult {
    pub states: Vec<DVector<f64>>,
    pub costates: Vec<DVector<f64>>,
    pub controls: Vec<DVector<f64>>,
    pub hamiltonians: Vec<f64>,
}

/// Hamiltonian for a continuous-time optimal control problem.
/// H(x, p, u) = p' * f(x, u) + L(x, u)
pub fn hamiltonian_value(
    x: &DVector<f64>,
    p: &DVector<f64>,
    u: &DVector<f64>,
    dynamics: &dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
    running_cost: &dyn Fn(&DVector<f64>, &DVector<f64>) -> f64,
) -> f64 {
    let f_val = dynamics(x, u);
    p.dot(&f_val) + running_cost(x, u)
}

/// For a linear system with quadratic cost, solve PMP via the Hamiltonian system.
pub fn solve_pmp_linear(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    x0: &DVector<f64>,
    t_final: f64,
    dt: f64,
) -> PmpResult {
    let r_inv = r.clone().try_inverse().expect("R not invertible");
    let brinvbt = b * &r_inv * b.transpose();

    let n = a.nrows();
    let n_steps = (t_final / dt).ceil() as usize;

    // Use DRE to get initial costate p(0) = P(0) * x0
    let dre = crate::riccati::solve_dre(a, b, q, r, &DMatrix::zeros(n, n), t_final, dt);
    let p0_init = if let Some((_, p_init)) = dre.last() {
        p_init * x0
    } else {
        DVector::zeros(n)
    };

    let mut states = Vec::with_capacity(n_steps + 1);
    let mut costates = Vec::with_capacity(n_steps + 1);
    let mut controls = Vec::with_capacity(n_steps);
    let mut hamiltonians = Vec::with_capacity(n_steps);

    let mut x = x0.clone();
    let mut p = p0_init;
    states.push(x.clone());
    costates.push(p.clone());

    for _ in 0..n_steps {
        let u = -&r_inv * b.transpose() * &p;

        let h_val = hamiltonian_value(
            &x, &p, &u,
            &|xi: &DVector<f64>, ui: &DVector<f64>| a * xi + b * ui,
            &|xi: &DVector<f64>, ui: &DVector<f64>| {
                let qx = q * xi;
                let ru = r * ui;
                xi.dot(&qx) + ui.dot(&ru)
            },
        );

        controls.push(u);
        hamiltonians.push(h_val);

        let dx = a * &x + b * controls.last().unwrap();
        let dp = -q * &x - a.transpose() * &p;

        x = x + dt * dx;
        p = p + dt * dp;

        states.push(x.clone());
        costates.push(p.clone());
    }

    PmpResult {
        states,
        costates,
        controls,
        hamiltonians,
    }
}

/// Verify Pontryagin conditions: check that H is minimized at u_opt.
pub fn verify_pmp_conditions(
    x: &DVector<f64>,
    p: &DVector<f64>,
    u_opt: &DVector<f64>,
    dynamics: &dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
    running_cost: &dyn Fn(&DVector<f64>, &DVector<f64>) -> f64,
    eps: f64,
) -> bool {
    let h_opt = hamiltonian_value(x, p, u_opt, dynamics, running_cost);

    for i in 0..u_opt.nrows() {
        let mut u_plus = u_opt.clone();
        u_plus[i] += eps;
        let h_plus = hamiltonian_value(x, p, &u_plus, dynamics, running_cost);

        let mut u_minus = u_opt.clone();
        u_minus[i] -= eps;
        let h_minus = hamiltonian_value(x, p, &u_minus, dynamics, running_cost);

        if h_opt > h_plus + 1e-8 || h_opt > h_minus + 1e-8 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_pmp_linear_1d() {
        let a = DMatrix::from_row_slice(1, 1, &[0.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let x0 = DVector::from_vec(vec![1.0]);

        let result = solve_pmp_linear(&a, &b, &q, &r, &x0, 5.0, 0.01);

        assert!(result.states.last().unwrap()[0].abs() < 1.0);
        assert!(result.costates.last().unwrap()[0].abs() < 0.5);
        for u in &result.controls {
            assert!(u[0].is_finite());
        }
    }

    #[test]
    fn test_pmp_hamiltonian_stationary() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let x0 = DVector::from_vec(vec![1.0, 0.0]);

        let result = solve_pmp_linear(&a, &b, &q, &r, &x0, 10.0, 0.01);

        for i in (500..result.states.len()).step_by(200) {
            if i < result.controls.len() {
                let q_c = q.clone();
                let r_c = r.clone();
                let a_c = a.clone();
                let b_c = b.clone();
                let verified = verify_pmp_conditions(
                    &result.states[i],
                    &result.costates[i],
                    &result.controls[i],
                    &|x: &DVector<f64>, u: &DVector<f64>| &a_c * x + &b_c * u,
                    &|x: &DVector<f64>, u: &DVector<f64>| {
                        (q_c.clone() * x).dot(x) + (r_c.clone() * u).dot(u)
                    },
                    0.01,
                );
                assert!(verified, "PMP conditions failed at step {}", i);
            }
        }
    }

    #[test]
    fn test_pmp_double_integrator() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.01]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let x0 = DVector::from_vec(vec![5.0, 0.0]);

        let result = solve_pmp_linear(&a, &b, &q, &r, &x0, 10.0, 0.01);

        assert!(result.states.last().unwrap()[0].abs() < 3.0);
        assert_eq!(result.states.len(), result.costates.len());
        assert_eq!(result.controls.len(), result.states.len() - 1);
    }

    #[test]
    fn test_verify_pmp_simple() {
        let x = DVector::from_vec(vec![1.0]);
        let p = DVector::from_vec(vec![2.0]);
        let u_opt = DVector::from_vec(vec![-1.0]);

        let verified = verify_pmp_conditions(
            &x, &p, &u_opt,
            &|xi: &DVector<f64>, ui: &DVector<f64>| {
                DVector::from_vec(vec![xi[0] + ui[0]])
            },
            &|xi: &DVector<f64>, ui: &DVector<f64>| {
                xi[0] * xi[0] + ui[0] * ui[0]
            },
            0.001,
        );
        assert!(verified);
    }

    #[test]
    fn test_verify_pmp_wrong_control() {
        let x = DVector::from_vec(vec![1.0]);
        let p = DVector::from_vec(vec![2.0]);
        let u_wrong = DVector::from_vec(vec![0.5]);

        let verified = verify_pmp_conditions(
            &x, &p, &u_wrong,
            &|xi: &DVector<f64>, ui: &DVector<f64>| {
                DVector::from_vec(vec![xi[0] + ui[0]])
            },
            &|xi: &DVector<f64>, ui: &DVector<f64>| {
                xi[0] * xi[0] + ui[0] * ui[0]
            },
            0.001,
        );
        assert!(!verified);
    }
}
