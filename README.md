# optimal-control

Optimal control in Rust. From LQR to HJB.

---

## What's Inside

| Module | What you get |
|---|---|
| **LQR** | Discrete and continuous LQR with gain matrix K and cost-to-go V(x) = x'Px |
| **Riccati** | DARE, CARE, and differential Riccati equation (finite horizon) |
| **Pontryagin** | Hamiltonian system solver, PMP condition verification |
| **HJB** | Grid-based 1D Hamilton-Jacobi-Bellman solver, LQR value function |
| **Bang-bang** | Double integrator time-optimal control, switching function analysis |
| **Trajectory** | Direct collocation and single shooting with gradient-based optimization |
| **Controllability** | Rank test, Gramian computation, stabilizability/detectability |
| **Dynamics** | Linear and nonlinear system simulation, Jacobian linearization |
| **Policy** | Policy computation, trajectory simulation, action planning, discrete action selection |

55 tests covering convergence, stability, and correctness across all modules.

## Install

```toml
[dependencies]
optimal-control = "0.1.0"
```

Requires **Rust 2021 edition**.

## Quick Start

### LQR for a discrete system

```rust
use optimal_control::lqr::solve_dlqr;
use nalgebra::{DMatrix, DVector};

let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
let q = DMatrix::identity(2, 2);
let r = DMatrix::from_row_slice(1, 1, &[0.1]);

let sol = solve_dlqr(&a, &b, &q, &r).unwrap();
let x = DVector::from_vec(vec![5.0, 1.0]);
let u = sol.control(&x);        // u = -Kx
let v = sol.cost_to_go(&x);     // V(x) = x'Px
```

### Controllability check

```rust
use optimal_control::controllability::check_controllability;

let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
let result = check_controllability(&a, &b);
assert!(result.controllable);
```

### Bang-bang control

```rust
use optimal_control::bangbang::bangbang_double_integrator;

let result = bangbang_double_integrator(5.0, 0.0, 1.0, 0.01);
```

## License

MIT OR Apache-2.0
