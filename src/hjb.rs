//! Hamilton-Jacobi-Bellman equation: dynamic programming approach to optimal control.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Grid-based HJB solver for 1D systems.
/// Solves the HJB equation: -∂V/∂t = min_u { L(x,u) + ∂V/∂x · f(x,u) }
/// using backward integration in time on a spatial grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HjbSolver1D {
    pub x_min: f64,
    pub x_max: f64,
    pub nx: usize,
    pub t_final: f64,
    pub nt: usize,
    /// Value function at each (time, space) grid point, stored as [t_idx * nx + x_idx]
    pub values: Vec<f64>,
}

impl HjbSolver1D {
    pub fn new(x_min: f64, x_max: f64, nx: usize, t_final: f64, nt: usize) -> Self {
        let values = vec![0.0; (nt + 1) * nx];
        Self { x_min, x_max, nx, t_final, nt, values }
    }

    pub fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.nx - 1) as f64
    }

    pub fn dt(&self) -> f64 {
        self.t_final / self.nt as f64
    }

    pub fn x_at(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx()
    }

    /// Solve the HJB equation with given dynamics and cost.
    pub fn solve(
        &mut self,
        dynamics: &dyn Fn(f64, f64) -> f64,        // f(x, u) -> dx/dt
        running_cost: &dyn Fn(f64, f64) -> f64,      // L(x, u) -> cost rate
        terminal_cost: &dyn Fn(f64) -> f64,           // psi(x) -> terminal cost
        u_min: f64,
        u_max: f64,
        nu: usize,
    ) {
        let dx = self.dx();
        let dt = self.dt();

        // Set terminal condition
        for i in 0..self.nx {
            let x = self.x_at(i);
            self.values[self.nt * self.nx + i] = terminal_cost(x);
        }

        // Backward time integration
                // Backward time integration
        let u_grid: Vec<f64> = (0..nu).map(|k| u_min + k as f64 * (u_max - u_min) / (nu - 1).max(1) as f64).collect();

        for k in (0..self.nt).rev() {
            for i in 0..self.nx {
                let x = self.x_at(i);

                // Spatial derivative using values at time k+1 (backward in time)
                let dvdx = if i == 0 {
                    (self.values[(k + 1) * self.nx + 1] - self.values[(k + 1) * self.nx]) / dx
                } else if i == self.nx - 1 {
                    (self.values[(k + 1) * self.nx + i] - self.values[(k + 1) * self.nx + i - 1]) / dx
                } else {
                    (self.values[(k + 1) * self.nx + i + 1] - self.values[(k + 1) * self.nx + i - 1]) / (2.0 * dx)
                };

                // Minimize over u: H = L(x,u) + dv/dx * f(x,u)
                let mut best_h = f64::INFINITY;
                for &u in &u_grid {
                    let h_val = running_cost(x, u) + dvdx * dynamics(x, u);
                    if h_val < best_h {
                        best_h = h_val;
                    }
                }

                // V(t,x) = V(t+dt,x) + dt * min_u H (HJB backward integration)
                let new_val = self.values[(k + 1) * self.nx + i] + dt * best_h;
                self.values[k * self.nx + i] = new_val;
            }
        }
    }

    /// Get the value function at time index k, position index i
    pub fn value_at(&self, k: usize, i: usize) -> f64 {
        self.values[k * self.nx + i]
    }

    /// Interpolate value at (t, x)
    pub fn interpolate(&self, t: f64, x: f64) -> f64 {
        let dt = self.dt();
        let dx = self.dx();

        let k_f = t / dt;
        let k = (k_f.floor() as usize).min(self.nt);
        let i_f = (x - self.x_min) / dx;
        let i = (i_f.floor() as usize).min(self.nx - 2).max(0) as usize;

        // Bilinear interpolation
        let t_alpha = k_f - k as f64;
        let x_alpha = i_f - i as f64;

        let k1 = (k + 1).min(self.nt);
        let i1 = (i + 1).min(self.nx - 1);

        let v00 = self.value_at(k, i);
        let v10 = self.value_at(k, i1);
        let v01 = self.value_at(k1, i);
        let v11 = self.value_at(k1, i1);

        let v0 = v00 * (1.0 - x_alpha) + v10 * x_alpha;
        let v1 = v01 * (1.0 - x_alpha) + v11 * x_alpha;

        v0 * (1.0 - t_alpha) + v1 * t_alpha
    }

    /// Get optimal control at (t, x) by checking which u minimizes H
    pub fn optimal_control(
        &self,
        t: f64,
        x: f64,
        dynamics: &dyn Fn(f64, f64) -> f64,
        running_cost: &dyn Fn(f64, f64) -> f64,
        u_min: f64,
        u_max: f64,
        nu: usize,
    ) -> f64 {
        let dt = self.dt();
        let dx = self.dx();

        let k = ((t / dt).floor() as usize).min(self.nt);
        let i_f = (x - self.x_min) / dx;
        let i = (i_f.floor() as usize).min(self.nx - 2).max(0) as usize;

        let dvdx = if i == 0 {
            (self.value_at(k.min(self.nt), 1) - self.value_at(k.min(self.nt), 0)) / dx
        } else if i >= self.nx - 2 {
            (self.value_at(k.min(self.nt), self.nx - 1) - self.value_at(k.min(self.nt), self.nx - 2)) / dx
        } else {
            (self.value_at(k.min(self.nt), i + 1) - self.value_at(k.min(self.nt), i - 1)) / (2.0 * dx)
        };

        let mut best_u = u_min;
        let mut best_val = f64::INFINITY;
        for j in 0..nu {
            let u = u_min + j as f64 * (u_max - u_min) / (nu - 1).max(1) as f64;
            let h = running_cost(x, u) + dvdx * dynamics(x, u);
            if h < best_val {
                best_val = h;
                best_u = u;
            }
        }
        best_u
    }
}

/// For linear-quadratic problems, the HJB value function has closed form:
/// V(t, x) = x' P(t) x where P solves the DRE.
pub fn hjb_value_lqr(
    x: &DVector<f64>,
    p: &DMatrix<f64>,
) -> f64 {
    let px = p * x;
    x.dot(&px)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_hjb_1d_double_integrator_like() {
        // dx/dt = u, cost = x^2 + u^2
        let mut solver = HjbSolver1D::new(-5.0, 5.0, 201, 3.0, 3000);
        solver.solve(
            &|_x, u| u,
            &|x, u| x * x + u * u,
            &|x| x * x,
            -3.0,
            3.0,
            61,
        );

        // Value at x=0 should be near 0
        let v_zero = solver.interpolate(0.0, 0.0);
        assert!(v_zero.abs() < 0.5, "V(0,0) should be near 0, got {}", v_zero);

        // Value at x=2 should be positive
        let v_two = solver.interpolate(0.0, 2.0);
        assert!(v_two > 0.1, "V(0,2) should be positive, got {}", v_two);

        // Optimal control near x=2 should be negative
        let u_opt = solver.optimal_control(0.0, 2.0, &|_x, u| u, &|x, u| x*x + u*u, -3.0, 3.0, 61);
        assert!(u_opt < 0.0, "Optimal control at x=2 should be negative, got {}", u_opt);
    }

    #[test]
    fn test_hjb_value_increases_with_state() {
        let mut solver = HjbSolver1D::new(-3.0, 3.0, 121, 2.0, 1000);
        solver.solve(
            &|_x, u| u,
            &|x, u| x * x + u * u,
            &|x| 10.0 * x * x,
            -3.0,
            3.0,
            61,
        );

        let v1 = solver.interpolate(0.0, 1.0);
        let v2 = solver.interpolate(0.0, 2.0);
        assert!(v2 > v1, "V should increase with |x|: V(1)={}, V(2)={}", v1, v2);
    }

    #[test]
    fn test_hjb_terminal_condition() {
        let mut solver = HjbSolver1D::new(-2.0, 2.0, 81, 1.0, 500);
        solver.solve(
            &|_x, u| u,
            &|_x, _u| 0.0,
            &|x| x * x,
            -1.0,
            1.0,
            21,
        );

        let x_test = 1.0;
        let v_terminal = solver.interpolate(1.0 - 1e-10, x_test);
        assert!((v_terminal - x_test * x_test).abs() < 0.2, "Terminal value mismatch: {} vs {}", v_terminal, x_test*x_test);
    }

    #[test]
    fn test_hjb_lqr_value() {
        let a = DMatrix::from_row_slice(1, 1, &[0.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);

        let p = crate::lqr::solve_care(&a, &b, &q, &r).unwrap();
        let x = DVector::from_vec(vec![1.0]);

        let v = hjb_value_lqr(&x, &p);
        assert!(v > 0.0, "LQR value should be positive");

        let expected = x.dot(&(p * &x));
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn test_hjb_control_direction() {
        let mut solver = HjbSolver1D::new(-5.0, 5.0, 201, 3.0, 3000);
        solver.solve(
            &|_x, u| u,
            &|x, u| x * x + u * u,
            &|x| x * x,
            -3.0, 3.0, 61,
        );

        let u_pos = solver.optimal_control(0.5, 1.0, &|_x, u| u, &|x, u| x*x + u*u, -3.0, 3.0, 61);
        assert!(u_pos < 0.0);

        let u_neg = solver.optimal_control(0.5, -1.0, &|_x, u| u, &|x, u| x*x + u*u, -3.0, 3.0, 61);
        assert!(u_neg > 0.0);
    }
}
