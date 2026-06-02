//! Bang-bang control: optimal controls that saturate at their limits.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Bang-bang control result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BangBangResult {
    pub states: Vec<DVector<f64>>,
    pub controls: Vec<f64>,
    pub times: Vec<f64>,
    pub switch_times: Vec<f64>,
}

/// For a double integrator (dx/dt = v, dv/dt = u, |u| <= u_max)
/// with minimum-time objective, the optimal control is bang-bang with at most
/// one switch. This is a classic problem.
///
/// Returns None if the target is not reachable.
pub fn bangbang_double_integrator(
    x0: f64,
    v0: f64,
    u_max: f64,
    dt: f64,
) -> BangBangResult {
    // Phase plane analysis for time-optimal bang-bang
    // Switching curve: v = -sign(x) * sqrt(2 * u_max * |x|)
    // If above curve: u = -u_max, if below: u = +u_max

    let mut states = Vec::new();
    let mut controls = Vec::new();
    let mut times = Vec::new();
    let mut switch_times = Vec::new();

    let mut x = x0;
    let mut v = v0;
    let mut t = 0.0;
    let mut switched = false;

    states.push(DVector::from_vec(vec![x, v]));
    times.push(t);

    for _ in 0..100000 {
        // Determine control direction
        let switching_val = if x >= 0.0 {
            v + (2.0 * u_max * x).sqrt()
        } else {
            v - (2.0 * u_max * (-x).abs()).sqrt()
        };

        let u = if switching_val > 0.0 {
            -u_max
        } else {
            u_max
        };

        controls.push(u);
        switch_times.clear();
        if switched {
            switch_times.push(t);
        }

        // Euler step
        v += u * dt;
        x += v * dt;
        t += dt;

        states.push(DVector::from_vec(vec![x, v]));
        times.push(t);

        // Check if we should switch (crossed switching curve)
        let new_switching_val = if x >= 0.0 {
            v + (2.0 * u_max * x).sqrt()
        } else {
            v - (2.0 * u_max * (-x).abs()).sqrt()
        };

        if !switched && new_switching_val.signum() != switching_val.signum() {
            switched = true;
            switch_times.push(t);
        }

        // Check convergence
        if x.abs() < 0.05 && v.abs() < 0.05 {
            break;
        }
    }

    BangBangResult {
        states,
        controls,
        times,
        switch_times,
    }
}

/// General bang-bang control for scalar control with bounds.
/// Uses the switching function: u* = u_max if sigma < 0, u_min if sigma > 0
/// where sigma is the switching function (typically the costate multiplied by B).
pub fn bangbang_control(
    switching_fn: &dyn Fn(f64) -> f64,  // sigma(t)
    u_min: f64,
    u_max: f64,
    times: &[f64],
) -> Vec<f64> {
    times.iter().map(|&t| {
        let sigma = switching_fn(t);
        if sigma > 0.0 {
            u_min
        } else if sigma < 0.0 {
            u_max
        } else {
            0.0 // Singular arc (rare)
        }
    }).collect()
}

/// Find switching times by finding zero crossings of the switching function.
pub fn find_switch_times(
    switching_fn: &dyn Fn(f64) -> f64,
    t_start: f64,
    t_end: f64,
    n_points: usize,
) -> Vec<f64> {
    let dt = (t_end - t_start) / n_points as f64;
    let mut switches = Vec::new();

    let mut prev_sign = switching_fn(t_start).signum();
    if prev_sign == 0.0 { prev_sign = 1.0; }
    for i in 1..=n_points {
        let t = t_start + i as f64 * dt;
        let val = switching_fn(t);
        let sign = val.signum();
        if sign == 0.0 { continue; }
        if sign != prev_sign {
            // Sign change detected, binary search
            let mut lo = t - dt;
            let mut hi = t;
            for _ in 0..50 {
                let mid = (lo + hi) / 2.0;
                let mid_sign = switching_fn(mid).signum();
                if mid_sign != prev_sign {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            switches.push((lo + hi) / 2.0);
            prev_sign = sign;
        }
    }

    switches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bangbang_double_integrator_from_rest() {
        let result = bangbang_double_integrator(5.0, 0.0, 1.0, 0.01);
        // Should reach near origin
        let final_state = result.states.last().unwrap();
        assert!(final_state[0].abs() < 0.1, "Final position should be near 0, got {}", final_state[0]);
        assert!(final_state[1].abs() < 0.1, "Final velocity should be near 0, got {}", final_state[1]);

        // Controls should be bang-bang (only +/- u_max)
        for &u in &result.controls {
            assert!((u - 1.0).abs() < 1e-10 || (u + 1.0).abs() < 1e-10,
                "Control should be ±u_max, got {}", u);
        }
    }

    #[test]
    fn test_bangbang_control_basic() {
        // Switching function crosses zero at t=1
        let controls = bangbang_control(
            &|t: f64| t - 1.0,
            -1.0,
            1.0,
            &[0.0, 0.5, 1.0, 1.5, 2.0],
        );
        assert_eq!(controls[0], 1.0);  // sigma(0) = -1 < 0 => u_max
        assert_eq!(controls[1], 1.0);  // sigma(0.5) = -0.5 < 0
        assert_eq!(controls[3], -1.0); // sigma(1.5) = 0.5 > 0 => u_min
        assert_eq!(controls[4], -1.0); // sigma(2) = 1 > 0
    }

    #[test]
    fn test_find_switch_times() {
        let switches = find_switch_times(&|t: f64| (t - 1.5) * (t - 3.0), 0.0, 5.0, 1000);
        assert_eq!(switches.len(), 2);
        assert!((switches[0] - 1.5).abs() < 0.01);
        assert!((switches[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_find_switch_times_single() {
        let switches = find_switch_times(&|t: f64| t - 2.0, 0.0, 4.0, 1000);
        assert_eq!(switches.len(), 1);
        assert!((switches[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_bangbang_reaches_target() {
        // Start at x=10, v=0
        let result = bangbang_double_integrator(10.0, 0.0, 2.0, 0.01);
        let final_x = result.states.last().unwrap()[0];
        assert!(final_x.abs() < 0.1, "Should reach near origin: {}", final_x);
    }

    #[test]
    fn test_bangbang_with_initial_velocity() {
        let result = bangbang_double_integrator(3.0, 2.0, 1.0, 0.01);
        let final_state = result.states.last().unwrap();
        assert!(final_state[0].abs() < 0.15);
        assert!(final_state[1].abs() < 0.15);
    }
}
