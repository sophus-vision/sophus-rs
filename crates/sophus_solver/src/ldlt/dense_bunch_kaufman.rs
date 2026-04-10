//! Bunch-Kaufman LDLᵀ factorization for symmetric indefinite matrices.
//!
//! Decomposes a symmetric (possibly indefinite) matrix A into:
//!
//! `P A Pᵀ = L D Lᵀ`
//!
//! where P is a permutation, L is unit lower-triangular, and D is block-diagonal
//! with 1×1 and 2×2 blocks. This handles the KKT saddle-point structure:
//!
//! ```text
//! [H + νI   Gᵀ]
//! [G       -εI]
//! ```
//!
//! Reference: Bunch & Kaufman (1977), "Some stable methods for calculating
//! inertia and solving symmetric linear systems".

use nalgebra::DMatrix;
use snafu::prelude::*;

/// Error during Bunch-Kaufman factorization.
#[derive(Debug, Clone, Snafu)]
pub enum BunchKaufmanError {
    /// Zero pivot encountered (matrix is singular).
    #[snafu(display("zero pivot at step {step}"))]
    ZeroPivot {
        /// Step at which the zero pivot occurred.
        step: usize,
    },
}

/// Result of Bunch-Kaufman LDLᵀ factorization.
///
/// Stores the factored form `P A Pᵀ = L D Lᵀ` where:
/// - L is unit lower-triangular (stored in the lower triangle of `ld`)
/// - D is block-diagonal with 1×1 and 2×2 blocks (stored on the diagonal/subdiagonal of `ld`)
/// - P is a permutation encoded by `pivot_indices`
#[derive(Debug, Clone)]
pub struct BunchKaufmanFactor {
    /// Combined L and D storage. Lower triangle holds L (unit diagonal not stored),
    /// diagonal and first subdiagonal hold D blocks.
    ld: DMatrix<f64>,
    /// Pivot indices. `pivot[k] >= 0` means 1×1 pivot at step k (row `pivot[k]`
    /// was swapped with row k). `pivot[k] < 0` means 2×2 pivot starting at step k
    /// (row `|pivot[k]| - 1` was swapped).
    pivot: Vec<i64>,
    /// Size of the matrix.
    n: usize,
}

/// Bunch-Kaufman threshold: α = (1 + √17) / 8 ≈ 0.6404.
const ALPHA: f64 = 0.6404;

/// Perform Bunch-Kaufman LDLᵀ factorization of a symmetric matrix.
///
/// The input matrix `a` is symmetric; only the lower triangle is read.
/// Returns the factored form.
pub fn factorize(a: &DMatrix<f64>) -> Result<BunchKaufmanFactor, BunchKaufmanError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());

    // Work on a copy — L and D are stored in-place.
    let mut ld = a.clone();
    // Symmetrise from lower to full (we need both triangles during elimination).
    for i in 0..n {
        for j in i + 1..n {
            ld[(i, j)] = ld[(j, i)];
        }
    }

    let mut pivot = vec![0i64; n];
    let mut k = 0usize;

    while k < n {
        // Determine pivot strategy.
        let a_kk = ld[(k, k)].abs();

        // Find largest off-diagonal magnitude in column k (below diagonal).
        let mut lambda = 0.0_f64;
        let mut r = k;
        for i in k + 1..n {
            let v = ld[(i, k)].abs();
            if v > lambda {
                lambda = v;
                r = i;
            }
        }

        if a_kk == 0.0 && lambda == 0.0 {
            // Zero column — singular but we can skip (set trivial pivot).
            pivot[k] = k as i64;
            k += 1;
            continue;
        }

        if a_kk >= ALPHA * lambda {
            // 1×1 pivot: use diagonal element.
            pivot[k] = k as i64;
            eliminate_1x1(&mut ld, k, n);
            k += 1;
        } else {
            // Check if we need 2×2 pivot.
            // Find sigma = max |A(i,r)| for i ≠ r, k ≤ i < n.
            let mut sigma = 0.0_f64;
            for i in k..n {
                if i != r {
                    sigma = sigma.max(ld[(i, r)].abs());
                }
            }

            if a_kk * sigma >= ALPHA * lambda * lambda {
                // 1×1 pivot is acceptable after all.
                pivot[k] = k as i64;
                eliminate_1x1(&mut ld, k, n);
                k += 1;
            } else if ld[(r, r)].abs() >= ALPHA * sigma {
                // 1×1 pivot using row r (swap r ↔ k).
                swap_rows_cols(&mut ld, k, r, n);
                pivot[k] = r as i64;
                eliminate_1x1(&mut ld, k, n);
                k += 1;
            } else {
                // 2×2 pivot using rows/cols k and r.
                if r != k + 1 {
                    swap_rows_cols(&mut ld, k + 1, r, n);
                }
                // Negative pivot index signals 2×2 pivot.
                pivot[k] = -(r as i64 + 1);
                pivot[k + 1] = -(r as i64 + 1);
                eliminate_2x2(&mut ld, k, n)?;
                k += 2;
            }
        }
    }

    Ok(BunchKaufmanFactor { ld, pivot, n })
}

/// Eliminate column k using a 1×1 pivot.
fn eliminate_1x1(ld: &mut DMatrix<f64>, k: usize, n: usize) {
    let d_kk = ld[(k, k)];
    if d_kk.abs() < 1e-30 {
        return; // Skip near-zero pivot.
    }
    let d_inv = 1.0 / d_kk;

    // Compute L[i,k] = A[i,k] / D[k,k] for i > k.
    // Update A[i,j] -= L[i,k] * D[k,k] * L[j,k] for i,j > k.
    for j in k + 1..n {
        let l_jk = ld[(j, k)] * d_inv;
        for i in j..n {
            ld[(i, j)] -= l_jk * ld[(i, k)];
            ld[(j, i)] = ld[(i, j)]; // Keep symmetric.
        }
        ld[(j, k)] = l_jk; // Store L[j,k].
    }
}

/// Eliminate columns k and k+1 using a 2×2 pivot.
fn eliminate_2x2(ld: &mut DMatrix<f64>, k: usize, n: usize) -> Result<(), BunchKaufmanError> {
    // D block = [d11 d21; d21 d22] at positions (k,k), (k+1,k), (k+1,k+1).
    let d11 = ld[(k, k)];
    let d21 = ld[(k + 1, k)];
    let d22 = ld[(k + 1, k + 1)];

    let det = d11 * d22 - d21 * d21;
    if det.abs() < 1e-30 {
        return Err(BunchKaufmanError::ZeroPivot { step: k });
    }
    let det_inv = 1.0 / det;

    // D⁻¹ = [d22, -d21; -d21, d11] / det
    for j in k + 2..n {
        let a_jk = ld[(j, k)];
        let a_jk1 = ld[(j, k + 1)];

        // L[j, k:k+1] = A[j, k:k+1] * D⁻¹
        let l_j0 = (d22 * a_jk - d21 * a_jk1) * det_inv;
        let l_j1 = (-d21 * a_jk + d11 * a_jk1) * det_inv;

        // Update A[i,j] -= [L[i,k] L[i,k+1]] * D * [L[j,k] L[j,k+1]]ᵀ
        for i in j..n {
            ld[(i, j)] -= ld[(i, k)] * l_j0 + ld[(i, k + 1)] * l_j1;
            ld[(j, i)] = ld[(i, j)];
        }

        ld[(j, k)] = l_j0;
        ld[(j, k + 1)] = l_j1;
    }

    Ok(())
}

/// Swap rows and columns p and q in a symmetric matrix.
fn swap_rows_cols(ld: &mut DMatrix<f64>, p: usize, q: usize, n: usize) {
    if p == q {
        return;
    }
    // Swap rows p and q.
    for j in 0..n {
        let tmp = ld[(p, j)];
        ld[(p, j)] = ld[(q, j)];
        ld[(q, j)] = tmp;
    }
    // Swap columns p and q.
    for i in 0..n {
        let tmp = ld[(i, p)];
        ld[(i, p)] = ld[(i, q)];
        ld[(i, q)] = tmp;
    }
}

impl BunchKaufmanFactor {
    /// Solve `A x = b` in-place on a raw slice.
    pub fn solve_slice_inplace(&self, x: &mut [f64]) {
        let n = self.n;
        assert_eq!(x.len(), n);

        // Apply permutation forward.
        let mut k = 0;
        while k < n {
            if self.pivot[k] >= 0 {
                let p = self.pivot[k] as usize;
                if p != k {
                    x.swap(k, p);
                }
                k += 1;
            } else {
                let p = (-self.pivot[k] - 1) as usize;
                if p != k + 1 {
                    x.swap(k + 1, p);
                }
                k += 2;
            }
        }

        // Forward solve: L y = P b.
        k = 0;
        while k < n {
            if self.pivot[k] >= 0 {
                for i in k + 1..n {
                    x[i] -= self.ld[(i, k)] * x[k];
                }
                k += 1;
            } else {
                for i in k + 2..n {
                    x[i] -= self.ld[(i, k)] * x[k] + self.ld[(i, k + 1)] * x[k + 1];
                }
                k += 2;
            }
        }

        // Diagonal solve: D z = y.
        k = 0;
        while k < n {
            if self.pivot[k] >= 0 {
                let d = self.ld[(k, k)];
                if d.abs() > 1e-30 {
                    x[k] /= d;
                }
                k += 1;
            } else {
                let d11 = self.ld[(k, k)];
                let d21 = self.ld[(k + 1, k)];
                let d22 = self.ld[(k + 1, k + 1)];
                let det = d11 * d22 - d21 * d21;
                if det.abs() > 1e-30 {
                    let t0 = x[k];
                    let t1 = x[k + 1];
                    x[k] = (d22 * t0 - d21 * t1) / det;
                    x[k + 1] = (-d21 * t0 + d11 * t1) / det;
                }
                k += 2;
            }
        }

        // Backward solve: Lᵀ w = z.
        k = n;
        while k > 0 {
            if k >= 2 && self.pivot[k - 2] < 0 {
                k -= 2;
                for i in k + 2..n {
                    x[k] -= self.ld[(i, k)] * x[i];
                    x[k + 1] -= self.ld[(i, k + 1)] * x[i];
                }
            } else {
                k -= 1;
                for i in k + 1..n {
                    x[k] -= self.ld[(i, k)] * x[i];
                }
            }
        }

        // Apply permutation backward.
        k = n;
        while k > 0 {
            if k >= 2 && self.pivot[k - 2] < 0 {
                k -= 2;
                let p = (-self.pivot[k] - 1) as usize;
                if p != k + 1 {
                    x.swap(k + 1, p);
                }
            } else {
                k -= 1;
                if self.pivot[k] >= 0 {
                    let p = self.pivot[k] as usize;
                    if p != k {
                        x.swap(k, p);
                    }
                }
            }
        }
    }
}
