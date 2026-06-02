//! Agent action optimization: best actions at each timestep.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::lqr::{self, LqrSolution};
use crate::controllability::{check_controllability, ControllabilityResult};

/// An agent's dynamics model: x_{k+1} = A x_k + B u_k
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentModel {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub state_labels: Vec<String>,
    pub action_labels: Vec<String>,
}

impl AgentModel {
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        state_labels: Vec<String>,
        action_labels: Vec<String>,
    ) -> Self {
        Self { a, b, state_labels, action_labels }
    }

    pub fn state_dim(&self) -> usize { self.a.nrows() }
    pub fn action_dim(&self) -> usize { self.b.ncols() }

    /// Check controllability
    pub fn controllability(&self) -> ControllabilityResult {
        check_controllability(&self.a, &self.b)
    }

    /// Compute optimal infinite-horizon policy (LQR)
    pub fn optimal_policy(
        &self,
        state_cost: &DMatrix<f64>,
        action_cost: &DMatrix<f64>,
    ) -> Result<AgentPolicy, String> {
        let sol = lqr::solve_dlqr(&self.a, &self.b, state_cost, action_cost)?;
        Ok(AgentPolicy {
            model: self.clone(),
            lqr: sol,
            state_cost: state_cost.clone(),
            action_cost: action_cost.clone(),
        })
    }
}

/// A computed policy for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPolicy {
    pub model: AgentModel,
    pub lqr: LqrSolution,
    pub state_cost: DMatrix<f64>,
    pub action_cost: DMatrix<f64>,
}

impl AgentPolicy {
    /// Get optimal action at a given state
    pub fn action(&self, state: &DVector<f64>) -> DVector<f64> {
        self.lqr.control(state)
    }

    /// Get value (cost-to-go) at a given state
    pub fn value(&self, state: &DVector<f64>) -> f64 {
        self.lqr.cost_to_go(state)
    }

    /// Get the gain matrix K
    pub fn gain(&self) -> &DMatrix<f64> {
        &self.lqr.k
    }

    /// Simulate the agent following this policy
    pub fn simulate(&self, x0: &DVector<f64>, n_steps: usize) -> AgentTrajectory {
        let mut states = Vec::with_capacity(n_steps + 1);
        let mut actions = Vec::with_capacity(n_steps);
        let mut values = Vec::with_capacity(n_steps + 1);

        states.push(x0.clone());
        values.push(self.value(x0));

        let mut x = x0.clone();
        for _ in 0..n_steps {
            let u = self.action(&x);
            x = &self.model.a * &x + &self.model.b * &u;
            states.push(x.clone());
            actions.push(u);
            values.push(self.value(&x));
        }

        AgentTrajectory { states, actions, values }
    }
}

/// A simulated trajectory of the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrajectory {
    pub states: Vec<DVector<f64>>,
    pub actions: Vec<DVector<f64>>,
    pub values: Vec<f64>,
}

impl AgentTrajectory {
    /// Total cumulative cost
    pub fn total_cost(&self) -> f64 {
        self.values.first().copied().unwrap_or(0.0)
    }

    /// Is the final state near zero (regulated)?
    pub fn is_regulated(&self, tolerance: f64) -> bool {
        self.states.last().map(|s| s.norm() < tolerance).unwrap_or(false)
    }

    /// Max action magnitude used
    pub fn max_action(&self) -> f64 {
        self.actions.iter().map(|u| u.norm()).fold(0.0_f64, f64::max)
    }
}

/// Multi-step action planner for reaching a target state
pub struct ActionPlanner {
    pub model: AgentModel,
    pub target: DVector<f64>,
}

impl ActionPlanner {
    pub fn new(model: AgentModel, target: DVector<f64>) -> Self {
        Self { model, target }
    }

    /// Plan actions to reach the target using LQR on the error system.
    pub fn plan(
        &self,
        x0: &DVector<f64>,
        state_cost: &DMatrix<f64>,
        action_cost: &DMatrix<f64>,
        terminal_weight: f64,
        n_steps: usize,
    ) -> AgentTrajectory {
        let sol = lqr::solve_dlqr(&self.model.a, &self.model.b, state_cost, action_cost)
            .expect("LQR solve failed");

        let mut states = Vec::with_capacity(n_steps + 1);
        let mut actions = Vec::with_capacity(n_steps);
        let mut values = Vec::with_capacity(n_steps + 1);

        let mut x = x0.clone();
        states.push(x.clone());

        let p = &sol.p;
        for _ in 0..n_steps {
            let e = &x - &self.target;
            let u = sol.control(&e);
            let pe = p * &e;
            let cost_to_go = e.dot(&pe);
            values.push(cost_to_go);
            x = &self.model.a * &x + &self.model.b * &u;
            states.push(x.clone());
            actions.push(u);
        }
        let e_final = x - &self.target;
        values.push(terminal_weight * e_final.dot(&e_final));

        AgentTrajectory { states, actions, values }
    }
}

/// Compute the best discrete action from a set of candidates.
pub fn best_discrete_action(
    state: &DVector<f64>,
    actions: &[DVector<f64>],
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    p: &DMatrix<f64>,
) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_cost = f64::INFINITY;

    for (idx, u) in actions.iter().enumerate() {
        let x_next = a * state + b * u;
        let state_cost = state.dot(&(q * state));
        let action_cost = u.dot(&(r * u));
        let future_cost = x_next.dot(&(p * &x_next));
        let total = state_cost + action_cost + future_cost;

        if total < best_cost {
            best_cost = total;
            best_idx = idx;
        }
    }

    (best_idx, best_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_agent() -> AgentModel {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        AgentModel::new(a, b, vec!["pos".into(), "vel".into()], vec!["force".into()])
    }

    #[test]
    fn test_agent_controllability() {
        let agent = make_simple_agent();
        let ctrl = agent.controllability();
        assert!(ctrl.controllable, "Simple agent should be controllable");
    }

    #[test]
    fn test_agent_policy_regulates() {
        let agent = make_simple_agent();
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let policy = agent.optimal_policy(&q, &r).unwrap();

        let x0 = DVector::from_vec(vec![5.0, 0.0]);
        let traj = policy.simulate(&x0, 200);

        assert!(traj.is_regulated(0.5), "Should regulate to near zero: norm = {}",
            traj.states.last().unwrap().norm());
    }

    #[test]
    fn test_agent_policy_value_positive() {
        let agent = make_simple_agent();
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let policy = agent.optimal_policy(&q, &r).unwrap();

        let x = DVector::from_vec(vec![1.0, 0.0]);
        assert!(policy.value(&x) > 0.0);
    }

    #[test]
    fn test_agent_trajectory_cost_decreasing() {
        let agent = make_simple_agent();
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let policy = agent.optimal_policy(&q, &r).unwrap();

        let x0 = DVector::from_vec(vec![3.0, 1.0]);
        let traj = policy.simulate(&x0, 100);

        assert!(traj.values[0] > traj.values[50]);
        assert!(traj.values[50] >= traj.values[100] - 0.01);
    }

    #[test]
    fn test_action_planner() {
        let agent = make_simple_agent();
        let target = DVector::from_vec(vec![1.0, 0.0]);
        let target_clone = target.clone();
        let planner = ActionPlanner::new(agent, target);

        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);

        let traj = planner.plan(&x0, &q, &r, 100.0, 200);

        let final_error = (traj.states.last().unwrap() - &target_clone).norm();
        assert!(final_error < 2.0, "Should approach target, error = {}", final_error);
    }

    #[test]
    fn test_best_discrete_action() {
        let a = DMatrix::from_row_slice(1, 1, &[1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[1.0]);
        let p = DMatrix::from_row_slice(1, 1, &[1.0]);

        let state = DVector::from_vec(vec![2.0]);
        let actions = vec![
            DVector::from_vec(vec![-2.0]),
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![2.0]),
        ];

        let (idx, _) = best_discrete_action(&state, &actions, &a, &b, &q, &r, &p);
        assert_eq!(idx, 0, "Should pick u=-2 to regulate state");
    }

    #[test]
    fn test_best_discrete_action_near_origin() {
        let a = DMatrix::from_row_slice(1, 1, &[0.9]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[10.0]);
        let p = DMatrix::from_row_slice(1, 1, &[1.0]);

        let state = DVector::from_vec(vec![0.1]);
        let actions = vec![
            DVector::from_vec(vec![-1.0]),
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
        ];

        let (idx, _) = best_discrete_action(&state, &actions, &a, &b, &q, &r, &p);
        assert_eq!(idx, 1, "Near origin with high action cost, should pick zero action");
    }

    #[test]
    fn test_agent_max_action() {
        let agent = make_simple_agent();
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.01]);
        let policy = agent.optimal_policy(&q, &r).unwrap();

        let x0 = DVector::from_vec(vec![10.0, 5.0]);
        let traj = policy.simulate(&x0, 50);

        assert!(traj.max_action().is_finite());
        assert!(traj.max_action() > 0.0);
    }
}
