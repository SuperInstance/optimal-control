//! # optimal-control
//!
//! Optimal control theory crate: LQR, Riccati, Pontryagin's Maximum Principle,
//! Hamilton-Jacobi-Bellman, trajectory optimization, controllability/observability,
//! and bang-bang control.

pub mod dynamics;
pub mod lqr;
pub mod riccati;
pub mod pontryagin;
pub mod hjb;
pub mod bangbang;
pub mod trajectory;
pub mod controllability;
pub mod agent;

pub use dynamics::*;
pub use lqr::*;
pub use riccati::*;
pub use pontryagin::*;
pub use hjb::*;
pub use bangbang::*;
pub use trajectory::*;
pub use controllability::*;
pub use agent::*;
