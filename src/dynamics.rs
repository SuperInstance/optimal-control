//! Controlled dynamical systems: state + control inputs.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A linear time-invariant controlled system: dx/dt = Ax + Bu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSystem {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
}

impl LinearSystem {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>) -> Self {
        assert_eq!(a.nrows(), a.ncols(), "A must be square");
        assert_eq!(a.nrows(), b.nrows(), "A and B must have same row count");
        Self { a, b }
    }

    pub fn state_dim(&self) -> usize {
        self.a.nrows()
    }

    pub fn control_dim(&self) -> usize {
        self.b.ncols()
    }

    /// Forward Euler step: x_{k+1} = x_k + dt * (A*x_k + B*u_k)
    pub fn step(&self, x: &DVector<f64>, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        x + dt * (&self.a * x + &self.b * u)
    }

    /// Simulate for n steps, returning all states.
    pub fn simulate(
        &self,
        x0: &DVector<f64>,
        dt: f64,
        n: usize,
        u_fn: &dyn Fn(&DVector<f64>, usize) -> DVector<f64>,
    ) -> Vec<DVector<f64>> {
        let mut states = Vec::with_capacity(n + 1);
        states.push(x0.clone());
        let mut x = x0.clone();
        for k in 0..n {
            let u = u_fn(&x, k);
            x = self.step(&x, &u, dt);
            states.push(x.clone());
        }
        states
    }
}

/// Nonlinear controlled system: dx/dt = f(x, u)
pub struct NonlinearSystem {
    pub f: Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>>,
    pub state_dim: usize,
    pub control_dim: usize,
}

impl NonlinearSystem {
    pub fn new(
        state_dim: usize,
        control_dim: usize,
        f: Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>>,
    ) -> Self {
        Self { state_dim, control_dim, f }
    }

    /// Forward Euler step
    pub fn step(&self, x: &DVector<f64>, u: &DVector<f64>, dt: f64) -> DVector<f64> {
        x + dt * (self.f)(x, u)
    }

    /// Simulate for n steps
    pub fn simulate(
        &self,
        x0: &DVector<f64>,
        dt: f64,
        n: usize,
        u_fn: &dyn Fn(&DVector<f64>, usize) -> DVector<f64>,
    ) -> Vec<DVector<f64>> {
        let mut states = Vec::with_capacity(n + 1);
        states.push(x0.clone());
        let mut x = x0.clone();
        for k in 0..n {
            let u = u_fn(&x, k);
            x = self.step(&x, &u, dt);
            states.push(x.clone());
        }
        states
    }

    /// Linearize around (x0, u0) using finite differences.
    pub fn linearize(&self, x0: &DVector<f64>, u0: &DVector<f64>, eps: f64) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = self.state_dim;
        let m = self.control_dim;
        let f0 = (self.f)(x0, u0);

        let mut a = DMatrix::zeros(n, n);
        for j in 0..n {
            let mut xp = x0.clone();
            xp[j] += eps;
            let mut xm = x0.clone();
            xm[j] -= eps;
            let fp = (self.f)(&xp, u0);
            let fm = (self.f)(&xm, u0);
            for i in 0..n {
                a[(i, j)] = (fp[i] - fm[i]) / (2.0 * eps);
            }
        }

        let mut b = DMatrix::zeros(n, m);
        for j in 0..m {
            let mut up = u0.clone();
            up[j] += eps;
            let mut um = u0.clone();
            um[j] -= eps;
            let fp = (self.f)(x0, &up);
            let fm = (self.f)(x0, &um);
            for i in 0..n {
                b[(i, j)] = (fp[i] - fm[i]) / (2.0 * eps);
            }
        }

        (a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_system_step() {
        let sys = LinearSystem::new(
            DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]),
            DMatrix::from_row_slice(2, 1, &[0.0, 1.0]),
        );
        let x = DVector::from_vec(vec![0.0, 1.0]);
        let u = DVector::from_vec(vec![0.5]);
        let x_next = sys.step(&x, &u, 0.1);
        // x_next = x + dt*(Ax + Bu) = [0,1] + 0.1*([1,0] + [0,0.5]) = [0,1] + [0.1, 0.05] = [0.1, 1.05]
        assert!((x_next[0] - 0.1).abs() < 1e-10);
        assert!((x_next[1] - 1.05).abs() < 1e-10);
    }

    #[test]
    fn test_linear_system_simulate() {
        let sys = LinearSystem::new(
            DMatrix::zeros(2, 2),
            DMatrix::identity(2, 2),
        );
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let states = sys.simulate(&x0, 0.1, 5, &|_x, _k| DVector::from_vec(vec![1.0, 0.0]));
        assert_eq!(states.len(), 6);
        // After 5 steps of u=[1,0]: x = [0.5, 0]
        assert!((states[5][0] - 0.5).abs() < 1e-10);
        assert!(states[5][1].abs() < 1e-10);
    }

    #[test]
    fn test_nonlinear_system_linearize() {
        let sys = NonlinearSystem::new(
            2, 1,
            Box::new(|x: &DVector<f64>, u: &DVector<f64>| {
                DVector::from_vec(vec![x[1], u[0]])
            }),
        );
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let u0 = DVector::from_vec(vec![0.0]);
        let (a, b) = sys.linearize(&x0, &u0, 1e-6);
        // f = [x2, u], so df/dx = [[0,1],[0,0]], df/du = [[0],[1]]
        assert!((a[(0, 1)] - 1.0).abs() < 1e-4);
        assert!((b[(1, 0)] - 1.0).abs() < 1e-4);
    }
}
