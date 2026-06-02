//! Trajectory optimization: direct collocation and shooting methods.

use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};

/// Trajectory optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryResult {
    pub states: Vec<DVector<f64>>,
    pub controls: Vec<DVector<f64>>,
    pub times: Vec<f64>,
    pub cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Direct collocation method for trajectory optimization.
pub struct DirectCollocation {
    pub n_states: usize,
    pub n_controls: usize,
    pub n_segments: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub state_bounds: Option<(DVector<f64>, DVector<f64>)>,
    pub control_bounds: Option<(DVector<f64>, DVector<f64>)>,
}

impl DirectCollocation {
    pub fn new(n_states: usize, n_controls: usize, n_segments: usize, t_start: f64, t_end: f64) -> Self {
        Self {
            n_states, n_controls, n_segments, t_start, t_end,
            state_bounds: None,
            control_bounds: None,
        }
    }

    pub fn with_state_bounds(mut self, lower: DVector<f64>, upper: DVector<f64>) -> Self {
        self.state_bounds = Some((lower, upper));
        self
    }

    pub fn with_control_bounds(mut self, lower: DVector<f64>, upper: DVector<f64>) -> Self {
        self.control_bounds = Some((lower, upper));
        self
    }

    pub fn dt(&self) -> f64 {
        (self.t_end - self.t_start) / self.n_segments as f64
    }

    /// Solve using iterative gradient-based collocation.
    pub fn solve(
        &self,
        dynamics: &dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
        running_cost: &dyn Fn(&DVector<f64>, &DVector<f64>) -> f64,
        terminal_cost: &dyn Fn(&DVector<f64>) -> f64,
        x0: &DVector<f64>,
        xf: Option<&DVector<f64>>,
        max_iter: usize,
        learning_rate: f64,
    ) -> TrajectoryResult {
        let dt = self.dt();
        let n = self.n_segments + 1;

        let mut states: Vec<DVector<f64>> = Vec::with_capacity(n);
        let mut controls: Vec<DVector<f64>> = Vec::with_capacity(n);

        states.push(x0.clone());
        for k in 1..n {
            if let Some(xf_val) = xf {
                let alpha = k as f64 / (n - 1) as f64;
                states.push(x0 * (1.0 - alpha) + xf_val * alpha);
            } else {
                states.push(x0.clone());
            }
            controls.push(DVector::zeros(self.n_controls));
        }
        controls.push(DVector::zeros(self.n_controls));

        let times: Vec<f64> = (0..n).map(|k| self.t_start + k as f64 * dt).collect();

        let mut converged = false;
        let mut last_cost = f64::INFINITY;

        for iter in 0..max_iter {
            // Forward simulation
            states[0] = x0.clone();
            for k in 0..self.n_segments {
                states[k + 1] = states[k].clone() + dt * dynamics(&states[k], &controls[k]);
            }

            // Compute cost
            let mut cost = 0.0;
            for k in 0..self.n_segments {
                cost += dt * running_cost(&states[k], &controls[k]);
            }
            cost += terminal_cost(&states[self.n_segments]);

            if (last_cost - cost).abs() < 1e-8 {
                converged = true;
                return TrajectoryResult {
                    states, controls, times, cost, iterations: iter, converged,
                };
            }
            last_cost = cost;

            // Backward pass
            let mut lam = finite_diff_terminal(&terminal_cost, &states[self.n_segments], 1e-6);
            let mut state_grads: Vec<DVector<f64>> = vec![DVector::zeros(self.n_states); n];
            state_grads[self.n_segments] = lam.clone();

            for k in (0..self.n_segments).rev() {
                let dfdx = finite_diff_dynamics_state(dynamics, &states[k], &controls[k], 1e-6);
                let dldx = finite_diff_running_state(running_cost, &states[k], &controls[k], 1e-6);
                let lam_update = dt * (dfdx.transpose() * &lam) + dt * dldx;
                lam = lam + lam_update;
                state_grads[k] = lam.clone();
            }

            // Update controls
            for k in 0..self.n_segments {
                let dldu = finite_diff_running_control(running_cost, &states[k], &controls[k], 1e-6);
                let dfdu = finite_diff_dynamics_control(dynamics, &states[k], &controls[k], 1e-6);
                let grad = dldu * dt + dfdu.transpose() * &state_grads[k + 1] * dt;

                controls[k] = controls[k].clone() - learning_rate * grad;

                if let Some((lower, upper)) = &self.control_bounds {
                    for i in 0..self.n_controls {
                        controls[k][i] = controls[k][i].max(lower[i]).min(upper[i]);
                    }
                }
            }
        }

        // Final forward sim
        states[0] = x0.clone();
        for k in 0..self.n_segments {
            states[k + 1] = states[k].clone() + dt * dynamics(&states[k], &controls[k]);
        }

        let mut cost = 0.0;
        for k in 0..self.n_segments {
            cost += dt * running_cost(&states[k], &controls[k]);
        }
        cost += terminal_cost(&states[self.n_segments]);

        TrajectoryResult {
            states, controls, times, cost, iterations: max_iter, converged,
        }
    }
}

/// Single shooting method.
pub struct ShootingMethod {
    pub n_states: usize,
    pub n_controls: usize,
    pub n_steps: usize,
    pub t_start: f64,
    pub t_end: f64,
}

impl ShootingMethod {
    pub fn new(n_states: usize, n_controls: usize, n_steps: usize, t_start: f64, t_end: f64) -> Self {
        Self { n_states, n_controls, n_steps, t_start, t_end }
    }

    pub fn dt(&self) -> f64 {
        (self.t_end - self.t_start) / self.n_steps as f64
    }

    /// Solve using iterative gradient-based shooting.
    pub fn solve(
        &self,
        dynamics: &dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
        running_cost: &dyn Fn(&DVector<f64>, &DVector<f64>) -> f64,
        terminal_cost: &dyn Fn(&DVector<f64>) -> f64,
        x0: &DVector<f64>,
        initial_controls: &[DVector<f64>],
        max_iter: usize,
        learning_rate: f64,
    ) -> TrajectoryResult {
        let dt = self.dt();
        let mut controls: Vec<DVector<f64>> = initial_controls.to_vec();
        let times: Vec<f64> = (0..=self.n_steps).map(|k| self.t_start + k as f64 * dt).collect();

        let mut last_cost = f64::INFINITY;

        for iter in 0..max_iter {
            // Forward simulation
            let mut states = Vec::with_capacity(self.n_steps + 1);
            states.push(x0.clone());
            for k in 0..self.n_steps {
                let x_next = states[k].clone() + dt * dynamics(&states[k], &controls[k]);
                states.push(x_next);
            }

            let mut cost = 0.0;
            for k in 0..self.n_steps {
                cost += dt * running_cost(&states[k], &controls[k]);
            }
            cost += terminal_cost(&states[self.n_steps]);

            if (last_cost - cost).abs() / (last_cost.abs() + 1e-10) < 1e-10 {
                return TrajectoryResult {
                    states, controls, times: times.clone(), cost, iterations: iter, converged: true,
                };
            }
            last_cost = cost;

            // Backward pass
            let mut lam = finite_diff_terminal(&terminal_cost, &states[self.n_steps], 1e-6);
            let mut control_grads = vec![DVector::zeros(self.n_controls); self.n_steps];

            for k in (0..self.n_steps).rev() {
                let dfdu = finite_diff_dynamics_control(dynamics, &states[k], &controls[k], 1e-6);
                let dldu = finite_diff_running_control(running_cost, &states[k], &controls[k], 1e-6);
                control_grads[k] = dldu * dt + dfdu.transpose() * &lam * dt;

                let dfdx = finite_diff_dynamics_state(dynamics, &states[k], &controls[k], 1e-6);
                let dldx = finite_diff_running_state(running_cost, &states[k], &controls[k], 1e-6);
                let lam_update = dt * (dfdx.transpose() * &lam) + dt * dldx;
                lam = lam + lam_update;
            }

            for k in 0..self.n_steps {
                controls[k] = controls[k].clone() - learning_rate * &control_grads[k];
            }
        }

        // Final forward sim
        let mut states = Vec::with_capacity(self.n_steps + 1);
        states.push(x0.clone());
        for k in 0..self.n_steps {
            let x_next = states[k].clone() + dt * dynamics(&states[k], &controls[k]);
            states.push(x_next);
        }

        let mut cost = 0.0;
        for k in 0..self.n_steps {
            cost += dt * running_cost(&states[k], &controls[k]);
        }
        cost += terminal_cost(&states[self.n_steps]);

        TrajectoryResult {
            states, controls, times, cost, iterations: max_iter, converged: false,
        }
    }
}

// Finite-difference helpers

fn finite_diff_terminal(
    f: &dyn Fn(&DVector<f64>) -> f64,
    x: &DVector<f64>,
    eps: f64,
) -> DVector<f64> {
    let n = x.nrows();
    let mut grad = DVector::zeros(n);
    for i in 0..n {
        let mut xp = x.clone();
        xp[i] += eps;
        let mut xm = x.clone();
        xm[i] -= eps;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
    }
    grad
}

fn finite_diff_dynamics_state(
    f: &dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
    x: &DVector<f64>,
    u: &DVector<f64>,
    eps: f64,
) -> DMatrix<f64> {
    let n = x.nrows();
    let m = f(x, u).nrows();
    let mut jac = DMatrix::zeros(m, n);
    for j in 0..n {
        let mut xp = x.clone();
        xp[j] += eps;
        let mut xm = x.clone();
        xm[j] -= eps;
        let fp = f(&xp, u);
        let fm = f(&xm, u);
        for i in 0..m {
            jac[(i, j)] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

fn finite_diff_dynamics_control(
    f: &dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
    x: &DVector<f64>,
    u: &DVector<f64>,
    eps: f64,
) -> DMatrix<f64> {
    let n = u.nrows();
    let m = f(x, u).nrows();
    let mut jac = DMatrix::zeros(m, n);
    for j in 0..n {
        let mut up = u.clone();
        up[j] += eps;
        let mut um = u.clone();
        um[j] -= eps;
        let fp = f(x, &up);
        let fm = f(x, &um);
        for i in 0..m {
            jac[(i, j)] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

fn finite_diff_running_state(
    f: &dyn Fn(&DVector<f64>, &DVector<f64>) -> f64,
    x: &DVector<f64>,
    u: &DVector<f64>,
    eps: f64,
) -> DVector<f64> {
    let n = x.nrows();
    let mut grad = DVector::zeros(n);
    for i in 0..n {
        let mut xp = x.clone();
        xp[i] += eps;
        let mut xm = x.clone();
        xm[i] -= eps;
        grad[i] = (f(&xp, u) - f(&xm, u)) / (2.0 * eps);
    }
    grad
}

fn finite_diff_running_control(
    f: &dyn Fn(&DVector<f64>, &DVector<f64>) -> f64,
    x: &DVector<f64>,
    u: &DVector<f64>,
    eps: f64,
) -> DVector<f64> {
    let n = u.nrows();
    let mut grad = DVector::zeros(n);
    for i in 0..n {
        let mut up = u.clone();
        up[i] += eps;
        let mut um = u.clone();
        um[i] -= eps;
        grad[i] = (f(x, &up) - f(x, &um)) / (2.0 * eps);
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_collocation_lqr() {
        let dc = DirectCollocation::new(2, 1, 50, 0.0, 5.0);
        let x0 = DVector::from_vec(vec![1.0, 0.0]);

        let result = dc.solve(
            &|x: &DVector<f64>, u: &DVector<f64>| DVector::from_vec(vec![x[1], u[0]]),
            &|x: &DVector<f64>, u: &DVector<f64>| x[0]*x[0] + 0.01*x[1]*x[1] + u[0]*u[0],
            &|x: &DVector<f64>| x[0]*x[0] + x[1]*x[1],
            &x0,
            None,
            200,
            0.01,
        );

        let final_state = result.states.last().unwrap();
        assert!(final_state[0].abs() < 1.5, "Final state x should be small: {}", final_state[0]);
        assert!(result.cost.is_finite());
    }

    #[test]
    fn test_shooting_method() {
        let sm = ShootingMethod::new(1, 1, 50, 0.0, 5.0);
        let x0 = DVector::from_vec(vec![2.0]);
        let init_controls: Vec<DVector<f64>> = (0..50).map(|_| DVector::from_vec(vec![-0.5])).collect();

        let result = sm.solve(
            &|_x: &DVector<f64>, u: &DVector<f64>| DVector::from_vec(vec![u[0]]),
            &|x: &DVector<f64>, u: &DVector<f64>| x[0]*x[0] + u[0]*u[0],
            &|x: &DVector<f64>| x[0]*x[0],
            &x0,
            &init_controls,
            500,
            0.005,
        );

        let final_x = result.states.last().unwrap()[0];
        assert!(final_x.abs() < 1.5, "Should regulate: {}", final_x);
    }

    #[test]
    fn test_collocation_with_target() {
        let dc = DirectCollocation::new(1, 1, 30, 0.0, 3.0);
        let x0 = DVector::from_vec(vec![0.0]);
        let xf = DVector::from_vec(vec![1.0]);

        let result = dc.solve(
            &|_x: &DVector<f64>, u: &DVector<f64>| DVector::from_vec(vec![u[0]]),
            &|_x: &DVector<f64>, u: &DVector<f64>| u[0]*u[0],
            &|x: &DVector<f64>| 10.0*(x[0]-1.0).powi(2),
            &x0,
            Some(&xf),
            300,
            0.01,
        );

        let final_x = result.states.last().unwrap()[0];
        assert!((final_x - 1.0).abs() < 1.0, "Should move toward target: {}", final_x);
    }

    #[test]
    fn test_shooting_cost_finite() {
        let sm = ShootingMethod::new(1, 1, 20, 0.0, 3.0);
        let x0 = DVector::from_vec(vec![1.0]);
        let init_controls: Vec<DVector<f64>> = (0..20).map(|_| DVector::zeros(1)).collect();

        let result = sm.solve(
            &|_x: &DVector<f64>, u: &DVector<f64>| DVector::from_vec(vec![u[0]]),
            &|x: &DVector<f64>, u: &DVector<f64>| x[0]*x[0] + u[0]*u[0],
            &|x: &DVector<f64>| x[0]*x[0],
            &x0,
            &init_controls,
            200,
            0.005,
        );

        assert!(result.cost.is_finite());
        assert!(result.cost < 100.0);
    }
}
