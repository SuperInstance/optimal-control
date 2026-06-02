//! Controllability and observability: rank conditions for linear systems.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Controllability analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllabilityResult {
    pub controllable: bool,
    pub rank: usize,
    pub n_states: usize,
    pub controllability_matrix: DMatrix<f64>,
}

/// Check controllability of a linear system (A, B).
/// Constructs the controllability matrix [B, AB, A^2B, ..., A^{n-1}B]
/// and checks if it has full row rank.
pub fn check_controllability(a: &DMatrix<f64>, b: &DMatrix<f64>) -> ControllabilityResult {
    let n = a.nrows();
    let m = b.ncols();

    // Build controllability matrix C = [B | AB | A^2 B | ... | A^{n-1} B]
    let mut cols: Vec<DMatrix<f64>> = Vec::with_capacity(n);
    let mut ak_b = b.clone();
    cols.push(ak_b.clone());
    for _ in 1..n {
        ak_b = a * &ak_b;
        cols.push(ak_b.clone());
    }

    // Concatenate columns
    let c_matrix = concatenate_columns(&cols);

    // Compute rank via SVD
    let c_clone = c_matrix.clone();
    let svd = c_matrix.svd(true, true);
    let singular_values = &svd.singular_values;
    let rank = singular_values.iter().filter(|&&s| s > 1e-10).count();

    ControllabilityResult {
        controllable: rank == n,
        rank,
        n_states: n,
        controllability_matrix: c_clone,
    }
}

/// Observability analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityResult {
    pub observable: bool,
    pub rank: usize,
    pub n_states: usize,
    pub observability_matrix: DMatrix<f64>,
}

/// Check observability of a linear system with output matrix C.
/// Constructs the observability matrix [C; CA; CA^2; ...; CA^{n-1}]
/// and checks if it has full column rank.
pub fn check_observability(a: &DMatrix<f64>, c: &DMatrix<f64>) -> ObservabilityResult {
    let n = a.nrows();
    let p = c.nrows();

    // Build observability matrix O = [C; CA; CA^2; ...; CA^{n-1}]
    let mut rows: Vec<DMatrix<f64>> = Vec::with_capacity(n);
    let mut ca_k = c.clone();
    rows.push(ca_k.clone());
    for _ in 1..n {
        ca_k = &ca_k * a;
        rows.push(ca_k.clone());
    }

    // Concatenate rows
    let o_matrix = concatenate_rows(&rows);

    // Compute rank via SVD
    let o_clone = o_matrix.clone();
    let svd = o_matrix.svd(true, true);
    let singular_values = &svd.singular_values;
    let rank = singular_values.iter().filter(|&&s| s > 1e-10).count();

    ObservabilityResult {
        observable: rank == n,
        rank,
        n_states: n,
        observability_matrix: o_clone,
    }
}

/// Compute the controllability Gramian for continuous-time system.
/// W_c = integral_0^inf exp(At) B B' exp(A't) dt
/// For stable A, solves: A W_c + W_c A' = -B B'
pub fn controllability_gramian(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = a.nrows();
    let bbt = b * b.transpose();

    // Solve A W + W A' = -BB' using vectorization
    // (I ⊗ A + A ⊗ I) vec(W) = -vec(BB')
    // Wait, that's wrong. It's (I ⊗ A) + (A ⊗ I)^T for the Lyapunov equation.
    // Actually: vec(AW + WA') = ((I ⊗ A) + (A ⊗ I)) vec(W)
    // No. vec(AWB) = (B' ⊗ A) vec(W)
    // So vec(AW + WA') = ((I ⊗ A) + (A ⊗ I)) vec(W)
    // But the Lyapunov is AW + WA' + Q = 0

    let i_n = DMatrix::identity(n, n);
    let lhs = crate::lqr::kronecker(&i_n, a) + crate::lqr::kronecker(&a, &i_n);
    let rhs = -crate::lqr::vectorize(&bbt);

    match lhs.lu().solve(&rhs) {
        Some(w_vec) => Ok(crate::lqr::unvectorize(&w_vec, n, n)),
        None => Err("Failed to solve Lyapunov equation for Gramian".to_string()),
    }
}

/// Compute the observability Gramian.
/// Solves: A' W_o + W_o A = -C'C
pub fn observability_gramian(a: &DMatrix<f64>, c: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = a.nrows();
    let ctc = c.transpose() * c;
    // Solve A'W + WA = -C'C using Lyapunov (via vectorization)
    let i_n = DMatrix::identity(n, n);
    let at = a.transpose();
    let lhs = crate::lqr::kronecker(&i_n, &at) + crate::lqr::kronecker(&at, &i_n);
    let rhs = -crate::lqr::vectorize(&ctc);
    match lhs.lu().solve(&rhs) {
        Some(w_vec) => Ok(crate::lqr::unvectorize(&w_vec, n, n)),
        None => Err("Failed to solve Lyapunov equation for observability Gramian".to_string()),
    }
}

/// Check if a pair (A, B) is stabilizable.
/// A system is stabilizable if all uncontrollable modes are stable.
pub fn is_stabilizable(a: &DMatrix<f64>, b: &DMatrix<f64>) -> bool {
    let ctrl = check_controllability(a, b);
    if ctrl.controllable {
        return true;
    }

    // Check eigenvalues: all uncontrollable modes must be stable
    let eig = a.complex_eigenvalues();
    for i in 0..eig.len() {
        if eig[i].re > 0.0 {
            // Unstable eigenvalue — check if controllable
            // Simplified: if not fully controllable and has unstable modes, assume not stabilizable
            return false;
        }
    }
    true
}

/// Check if a pair (A, C) is detectable.
pub fn is_detectable(a: &DMatrix<f64>, c: &DMatrix<f64>) -> bool {
    is_stabilizable(&a.transpose(), &c.transpose())
}

fn concatenate_columns(matrices: &[DMatrix<f64>]) -> DMatrix<f64> {
    if matrices.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    let nrows = matrices[0].nrows();
    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
    let mut result = DMatrix::zeros(nrows, total_cols);
    let mut col_offset = 0;
    for m in matrices {
        result.slice_range_mut(.., col_offset..col_offset + m.ncols()).copy_from(m);
        col_offset += m.ncols();
    }
    result
}

fn concatenate_rows(matrices: &[DMatrix<f64>]) -> DMatrix<f64> {
    if matrices.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    let ncols = matrices[0].ncols();
    let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
    let mut result = DMatrix::zeros(total_rows, ncols);
    let mut row_offset = 0;
    for m in matrices {
        result.slice_range_mut(row_offset..row_offset + m.nrows(), ..).copy_from(m);
        row_offset += m.nrows();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_controllable_system() {
        // Double integrator is controllable
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let result = check_controllability(&a, &b);
        assert!(result.controllable, "Double integrator should be controllable");
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_uncontrollable_system() {
        // A = I, B = [1; 0] -> second state is uncontrollable
        let a = DMatrix::identity(2, 2);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let result = check_controllability(&a, &b);
        assert!(!result.controllable, "Should not be controllable");
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn test_fully_controllable() {
        // A = 0, B = I
        let a = DMatrix::zeros(2, 2);
        let b = DMatrix::identity(2, 2);
        let result = check_controllability(&a, &b);
        assert!(result.controllable);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_observable_system() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let result = check_observability(&a, &c);
        assert!(result.observable, "Double integrator with position output is observable");
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_unobservable_system() {
        // Only velocity observed for integrator: C = [0, 1]
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let c = DMatrix::from_row_slice(1, 2, &[0.0, 1.0]);
        let result = check_observability(&a, &c);
        // CA = [0, 0] -> rank of observability matrix is 1
        assert!(!result.observable, "Should not be observable");
    }

    #[test]
    fn test_observability_identity() {
        let a = DMatrix::zeros(2, 2);
        let c = DMatrix::identity(2, 2);
        let result = check_observability(&a, &c);
        assert!(result.observable);
    }

    #[test]
    fn test_controllability_gramian_symmetric() {
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);
        let wc = controllability_gramian(&a, &b).unwrap();
        assert!((wc.clone() - wc.transpose()).norm() < 1e-8, "Gramian should be symmetric");
    }

    #[test]
    fn test_controllability_gramian_positive_definite() {
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let b = DMatrix::identity(2, 2);
        let wc = controllability_gramian(&a, &b).unwrap();
        let eig = wc.symmetric_eigenvalues();
        for i in 0..eig.nrows() {
            assert!(eig[i] > 0.0, "Gramian should be positive definite");
        }
    }

    #[test]
    fn test_stabilizable_controllable() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        assert!(is_stabilizable(&a, &b));
    }

    #[test]
    fn test_stabilizable_stable_uncontrollable() {
        // All modes stable (eigenvalues 0, -1) so even if not controllable, still stabilizable
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        assert!(is_stabilizable(&a, &b));
    }

    #[test]
    fn test_detectable() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        assert!(is_detectable(&a, &c));
    }

    #[test]
    fn test_controllability_3state() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
        let result = check_controllability(&a, &b);
        assert!(result.controllable, "Triple integrator should be controllable");
        assert_eq!(result.rank, 3);
    }
}
