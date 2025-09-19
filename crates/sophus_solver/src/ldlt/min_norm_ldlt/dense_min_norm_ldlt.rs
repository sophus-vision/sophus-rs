use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    ldlt::{
        DenseLdltFactor,
        min_norm_ldlt::{
            IsMinNormLdltBackend,
            MinNormLdlt,
        },
    },
    matrix::{
        BlockRange,
        PartitionBlockIndex,
    },
};

// /// This is a min-norm solver for a positive semi-definite linear system.
// ///
// /// Let us assume we have a linear system `A x = b` and we factorized `A` using
// /// the LDLt decomposition.
// ///
// /// If A has full rank then solving X for `A X = I` produces the inverse of `A`.
// ///
// ///
// /// If rank(A) < n, then A is not invertible. However, in many cases we seek to construct the
// /// (Moore-Penrose) pseudo-inverse of A instead. It is important to note
// /// that the `A X = I` won't lead to pseudo-inverse in general, if A is rank deficient. In
// /// particular, `A x = b` has multiple solutions.
// ///
// /// min |X| A X = I.
// #[derive(Clone, Debug)]
// pub struct MinNormPsd {
//     ldlt: DenseLdltFactor,
//     gram_schmidt: Option<GramSchmidt>,
//     mat_rd: nalgebra::DMatrix<f64>,
//     mat_dr: nalgebra::DMatrix<f64>,
// }

// impl MinNormPsd {
//     /// f
//     pub fn from(ldlt: DenseLdltFactor) -> Self {
//         let n = ldlt.mat_l.scalar_dimension();

//         let mut gram_schmidt = None;
//         if ldlt.rank < n {
//             let positive_diag_indices: Vec<usize> =
//                 (0..n).filter(|&t| ldlt.diag_d[t] > 0.0).collect();
//             assert_eq!(positive_diag_indices.len(), ldlt.rank);
//             // Gather D_J (the positive pivots from the original factorization)
//             let mut positive_diag_values = nalgebra::DVector::<f64>::zeros(ldlt.rank);
//             for (c, &t) in positive_diag_indices.iter().enumerate() {
//                 positive_diag_values[c] = ldlt.diag_d[t];
//             }

//             // --- Build G = L_{[:,J]}^T L_{[:,J]} (r×r). (Consider caching per factorization.)
//             let mut gram_mat_l = nalgebra::DMatrix::<f64>::zeros(ldlt.rank, ldlt.rank);
//             let mat_l = ldlt.mat_l.view();
//             for a in 0..ldlt.rank {
//                 let ta = positive_diag_indices[a];
//                 for b in 0..=a {
//                     let tb = positive_diag_indices[b];
//                     let start = ta.max(tb);
//                     let mut s = 0.0;
//                     for row in start..n {
//                         s += mat_l[(row, ta)] * mat_l[(row, tb)];
//                     }
//                     gram_mat_l[(a, b)] = s;
//                     gram_mat_l[(b, a)] = s;
//                 }
//             }
//             let mut gram_diag_d = nalgebra::DVector::<f64>::zeros(ldlt.rank);
//             // G is SPD over positive pivots
//             ldlt_decompose_inplace(
//                 gram_mat_l.as_view_mut(),
//                 gram_diag_d.as_view_mut(),
//                 ldlt.tol_rel,
//             )
//             .unwrap();
//             gram_schmidt = Some(GramSchmidt {
//                 positive_diag_indices,
//                 gram_diag_d,
//                 positive_diag_values,
//                 gram_mat_l,
//             })
//         }

//         let max_dim = ldlt
//             .mat_l
//             .partitions()
//             .iter()
//             .map(|s| s.block_dimension)
//             .max()
//             .unwrap();
//         let mat_rd = nalgebra::DMatrix::<f64>::zeros(ldlt.rank, max_dim);
//         let mat_dr = nalgebra::DMatrix::<f64>::zeros(max_dim, ldlt.rank);

//         MinNormPsd {
//             gram_schmidt,
//             mat_rd,
//             mat_dr,
//             ldlt,
//         }
//     }
// }

// impl IsMinNormFactor for MinNormPsd {
//     fn inverse(&self) -> nalgebra::DMatrix<f64> {
//         let n = self.ldlt.mat_l.scalar_dimension();
//         let r = self.ldlt.rank;

//         // Handle edge cases early.
//         if n == 0 || r == 0 {
//             return DMatrix::<f64>::zeros(n, n);
//         }

//         if let Some(gram_schmidt) = &self.gram_schmidt {
//             // 1) Gather E = L_{:,J}  (n × r)
//             let mat_l = self.ldlt.mat_l.view();
//             let mut e = DMatrix::<f64>::zeros(n, r);
//             for (c, &t) in gram_schmidt.positive_diag_indices.iter().enumerate() {
//                 // L is unit-lower: entries exist only for row >= t; above-diagonal is zero.
//                 for row in t..n {
//                     e[(row, c)] = mat_l[(row, t)];
//                 }
//             }

//             // 2) A = E * G^{-1} * D_J^{-1} * G^{-1}  (n × r)
//             let mut a = e.clone();
//             // right-solve by G twice with a diagonal in between
//             ldlt_right_matsolve_inplace(
//                 gram_schmidt.gram_mat_l.as_view(),
//                 gram_schmidt.gram_diag_d.as_view(),
//                 a.as_view_mut(),
//             );
//             diag_right_matsolve_inplace(
//                 gram_schmidt.positive_diag_values.as_view(),
//                 &mut a.as_view_mut(),
//             );
//             ldlt_right_matsolve_inplace(
//                 gram_schmidt.gram_mat_l.as_view(),
//                 gram_schmidt.gram_diag_d.as_view(),
//                 a.as_view_mut(),
//             );

//             // 3) S = A * Eᵀ  (n × n)
//             let mut out = DMatrix::<f64>::zeros(n, n);
//             let e_t = e.transpose(); // (r × n)
//             a.mul_to(&e_t, &mut out);

//             return out;
//         }

//         // and here

//         // H = L D Lᵀ  =>  H^{-1} = L^{-T} D^{-1} L^{-1}
//         let mat_l = self.ldlt.mat_l.view();

//         // Step 1: W = L^{-1} I  (forward substitution on identity)
//         let mut w = DMatrix::<f64>::identity(n, n);
//         lower_matsolve_inplace(&mat_l, &mut w.as_view_mut());

//         // Step 2: W <- D^{-1} W  (row scaling by the pivots)
//         // In the SPD case, positive_diag_values has length n and equals diag(D) in order.
//         diag_matsolve_inplaced(self.ldlt.diag_d.as_view(), &mut w.as_view_mut());

//         // Step 3: Z = L^{-T} W  (back substitution)
//         lower_transpose_matsolve_inplace(&mat_l, &mut w.as_view_mut());

//         w // full dense inverse
//     }

//     fn get_inv_block(
//         &mut self,
//         row_idx: PartitionBlockIndex,
//         col_idx: PartitionBlockIndex,
//     ) -> nalgebra::DMatrix<f64> {
//         let n = self.ldlt.mat_l.scalar_dimension();
//         debug_assert_eq!(n, self.ldlt.diag_d.len());

//         let (si, di) = self.ldlt.mat_l.block_range(row_idx);
//         let (sj, dj) = self.ldlt.mat_l.block_range(col_idx);

//         let mut out = nalgebra::DMatrix::<f64>::zeros(di, dj);

//         let mat_l = self.ldlt.mat_l.view();

//         // Positive pivots J and rank r
//         if self.ldlt.rank == 0 {
//             return out;
//         }

//         if let Some(gram_schmidt) = &self.gram_schmidt {
//             if dj <= di {
//                 // -------- small-on-right (your original path, but with row-scaling helper)
//                 // B := L_{jJ}^T (r×dj)
//                 let mut b = self.mat_rd.view_mut((0, 0), (self.ldlt.rank, dj));

//                 for (c, &t) in gram_schmidt.positive_diag_indices.iter().enumerate() {
//                     for k in 0..dj {
//                         b[(c, k)] = mat_l[(sj + k, t)];
//                     }
//                 }

//                 // W = G^{-1} L_{jJ}^T
//                 ldlt_matsolve_inplace(
//                     gram_schmidt.gram_mat_l.as_view(),
//                     gram_schmidt.gram_diag_d.as_view(),
//                     b.as_view_mut(),
//                 );
//                 // Row-scale by D_J^{-1}
//                 diag_matsolve_inplaced(
//                     gram_schmidt.positive_diag_values.as_view(),
//                     &mut b.as_view_mut(),
//                 );
//                 // U = G^{-1} W
//                 ldlt_matsolve_inplace(
//                     gram_schmidt.gram_mat_l.as_view(),
//                     gram_schmidt.gram_diag_d.as_view(),
//                     b.as_view_mut(),
//                 );

//                 // A := L_{iJ} (di×r)
//                 let mut a = self.mat_dr.view_mut((0, 0), (di, self.ldlt.rank));

//                 for (c, &t) in gram_schmidt.positive_diag_indices.iter().enumerate() {
//                     for k in 0..di {
//                         a[(k, c)] = mat_l[(si + k, t)];
//                     }
//                 }

//                 a.mul_to(&b, &mut out); // di × dj

//                 return out;
//             } else {
//                 // -------- small-on-left (apply M from the right to L_{iJ})
//                 // A := L_{iJ} (di×r)
//                 let mut a = self.mat_dr.view_mut((0, 0), (di, self.ldlt.rank));

//                 for (c, &t) in gram_schmidt.positive_diag_indices.iter().enumerate() {
//                     for k in 0..di {
//                         a[(k, c)] = mat_l[(si + k, t)];
//                     }
//                 }

//                 // A <- A * G^{-1} * D_J^{-1} * G^{-1}
//                 ldlt_right_matsolve_inplace(
//                     gram_schmidt.gram_mat_l.as_view(),
//                     gram_schmidt.gram_diag_d.as_view(),
//                     a.as_view_mut(),
//                 ); // * G^{-1}
//                 diag_right_matsolve_inplace(
//                     gram_schmidt.positive_diag_values.as_view(),
//                     &mut a.as_view_mut(),
//                 ); // * D_J^{-1}
//                 ldlt_right_matsolve_inplace(
//                     gram_schmidt.gram_mat_l.as_view(),
//                     gram_schmidt.gram_diag_d.as_view(),
//                     a.as_view_mut(),
//                 ); // * G^{-1}

//                 // S = A * L_{jJ}^T
//                 let mut b = self.mat_rd.view_mut((0, 0), (self.ldlt.rank, dj));
//                 for (c, &t) in gram_schmidt.positive_diag_indices.iter().enumerate() {
//                     for k in 0..dj {
//                         b[(c, k)] = mat_l[(sj + k, t)];
//                     }
//                 }
//                 a.mul_to(&b, &mut out); // di × dj
//             }
//             return out;
//         }

//         // Compute the (si..si+di, sj..sj+dj) block of H^{-1} = L^{-T} D^{-1} L^{-1}
//         // by forming the selected columns of H^{-1} via:
//         //   1) solve L * X = E_j     (forward substitution, E_j selects the j-block columns)
//         //   2) X <- D^{-1} X         (row scaling)
//         //   3) solve Lᵀ * Z = X      (back substitution)  => Z = H^{-1}[:, cols_j]
//         // then extract rows_i from Z.

//         let n = self.ldlt.mat_l.scalar_dimension();
//         let (si, di) = self.ldlt.mat_l.block_range(row_idx);
//         let (sj, dj) = self.ldlt.mat_l.block_range(col_idx);

//         let mat_l = self.ldlt.mat_l.view();

//         // Scratch: n × dj (we reuse mat_rd; here r == n so it has enough rows)
//         let mut w = self.mat_rd.view_mut((0, 0), (n, dj));

//         // Initialize with B = E_j (identity columns for the selected block)
//         // First, zero everything.
//         for i in 0..n {
//             for k in 0..dj {
//                 w[(i, k)] = 0.0;
//             }
//         }
//         // Set the identity columns in the selected j-block.
//         for k in 0..dj {
//             w[(sj + k, k)] = 1.0;
//         }

//         // 1) Forward solve: L * W = E_j  => W = L^{-1} E_j
//         lower_matsolve_inplace(&mat_l, &mut w.as_view_mut());

//         // 2) Row-scale by D^{-1}: W <- D^{-1} W
//         // For full rank, positive_diag_values has length n and equals diag(D) in order.
//         diag_matsolve_inplaced(self.ldlt.diag_d.as_view(), &mut w.as_view_mut());

//         // 3) Backward solve: Lᵀ * Z = W  => Z = L^{-T} W
//         lower_transpose_matsolve_inplace(&mat_l, &mut w.as_view_mut());

//         // Extract the requested block: rows si..si+di, columns correspond to the j-block (0..dj
// in         // W)
//         for rr in 0..di {
//             for cc in 0..dj {
//                 out[(rr, cc)] = w[(si + rr, cc)];
//             }
//         }

//         out
//     }
// }

/// f
#[derive(Clone, Debug)]
pub struct DenseBackend {
    ldlt: DenseLdltFactor,
    positive_pivot_idx: Vec<usize>,
    positive_pivot_values: DVector<f64>,
}

impl IsMinNormLdltBackend for DenseBackend {
    type LdltFactor = DenseLdltFactor;

    fn new(ldlt: Self::LdltFactor) -> Self {
        let n = ldlt.mat_l.scalar_dimension();
        let positive_pivot_idx: Vec<usize> = (0..n).filter(|&t| ldlt.diag_d[t] > 0.0).collect();
        let mut positive_pivot_values = DVector::<f64>::zeros(positive_pivot_idx.len());
        for (c, &t) in positive_pivot_idx.iter().enumerate() {
            positive_pivot_values[c] = ldlt.diag_d[t];
        }
        DenseBackend {
            ldlt,
            positive_pivot_idx,
            positive_pivot_values,
        }
    }

    fn scalar_dim(&self) -> usize {
        self.ldlt.mat_l.scalar_dimension()
    }

    fn tol_rel(&self) -> f64 {
        self.ldlt.tol_rel
    }

    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]) {
        // L is dense unit-lower with stored diagonal.
        let n = self.scalar_dim();
        let l = self.ldlt.mat_l.view();
        for i in 0..col_j {
            out[i] = 0.0;
        }
        for i in col_j..n {
            out[i] = l[(i, col_j)];
        } // includes out[t] = 1.0
    }

    fn try_column_of_inverse(&self, col_range: &BlockRange) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        use crate::kernel::{
            diag_matsolve_inplaced,
            lower_matsolve_inplace,
            lower_transpose_matsolve_inplace,
        };

        let n = self.scalar_dim();
        debug_assert!(col_range.start_idx + col_range.block_dim <= n);

        // Build RHS = E_j (n × dj), i.e., selected identity columns.
        let mut w = DMatrix::<f64>::zeros(n, col_range.block_dim);
        for k in 0..col_range.block_dim {
            w[(col_range.start_idx + k, k)] = 1.0;
        }

        // H^{-1}[:, sj..sj+dj] = L^{-T} D^{-1} L^{-1} E_j
        let l = self.ldlt.mat_l.view();
        lower_matsolve_inplace(&l, &mut w.as_view_mut()); // W = L^{-1} E_j
        diag_matsolve_inplaced(self.ldlt.diag_d.as_view(), &mut w.as_view_mut()); // W = D^{-1} W
        lower_transpose_matsolve_inplace(&l, &mut w.as_view_mut()); // W = L^{-T} W
        Some(w)
    }

    fn try_inverse(&self) -> Option<DMatrix<f64>> {
        if self.rank() != self.scalar_dim() {
            return None;
        }
        // H^{-1} = L^{-T} D^{-1} L^{-1} using your kernels
        use crate::kernel::{
            diag_matsolve_inplaced,
            lower_matsolve_inplace,
            lower_transpose_matsolve_inplace,
        };
        let n = self.scalar_dim();
        let l = self.ldlt.mat_l.view();
        let mut w = DMatrix::<f64>::identity(n, n);
        lower_matsolve_inplace(&l, &mut w.as_view_mut());
        diag_matsolve_inplaced(self.ldlt.diag_d.as_view(), &mut w.as_view_mut());
        lower_transpose_matsolve_inplace(&l, &mut w.as_view_mut());
        Some(w)
    }

    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.ldlt.mat_l.partitions().block_range(idx)
    }

    fn positive_pivot_idx(&self) -> &[usize] {
        &self.positive_pivot_idx
    }

    fn positive_pivot_values(&self) -> &DVector<f64> {
        &self.positive_pivot_values
    }
}

/// g
pub type DenseMinNormFactor = MinNormLdlt<DenseBackend>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{
        Cholesky,
        DMatrix,
    };

    use crate::{
        IsLinearSolver,
        IsMinNormFactor,
        ldlt::{
            DenseLdlt,
            IntoMinNormPsd,
        },
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            dense::{
                DenseSymmetricMatrix,
                DenseSymmetricMatrixBuilder,
            },
        },
        svd::pseudo_inverse_with_tol,
    };

    const LDLT_TOL: f64 = 1e-12;

    fn build_square_from_dense(
        partitions: PartitionSet,
        h_dense: &DMatrix<f64>,
    ) -> DenseSymmetricMatrix {
        // Compute region offsets
        let mut region_offsets = Vec::with_capacity(partitions.len());
        let mut off = 0usize;
        for p in partitions.specs() {
            region_offsets.push(off);
            off += p.block_count * p.block_dim;
        }
        assert_eq!(off, h_dense.nrows());
        assert_eq!(off, h_dense.ncols());

        let mut b = DenseSymmetricMatrixBuilder::zero(partitions.clone());

        for partition_idx in 0..partitions.len() {
            let pi = &partitions.specs()[partition_idx];
            let oi = region_offsets[partition_idx];

            for block_row_idx in 0..pi.block_count {
                let di = pi.block_dim;
                let si = oi + block_row_idx * di;

                // 1) diagonal block inside region ri
                let idx = PartitionBlockIndex {
                    partition: partition_idx,
                    block: block_row_idx,
                };
                b.add_lower_block(idx, idx, &h_dense.view((si, si), (di, di)));

                // 2) within-region lower off-diagonals (same region, earlier block index)
                for block_col_idx in 0..block_row_idx {
                    let dj = pi.block_dim;
                    let sj = oi + block_col_idx * dj;
                    // lower triangle -> (bi,bj) with bj < bi
                    b.add_lower_block(
                        idx,
                        PartitionBlockIndex {
                            partition: partition_idx,
                            block: block_col_idx,
                        },
                        &h_dense.view((si, sj), (di, dj)),
                    );
                }

                // 3) cross-region lower off-diagonals (regions rj < ri)
                for rj in 0..partition_idx {
                    let pj = &partitions.specs()[rj];
                    let oj = region_offsets[rj];
                    for bj in 0..pj.block_count {
                        let dj = pj.block_dim;
                        let sj = oj + bj * dj;
                        b.add_lower_block(
                            idx,
                            PartitionBlockIndex {
                                partition: rj,
                                block: bj,
                            },
                            &h_dense.view((si, sj), (di, dj)),
                        );
                    }
                }
            }
        }

        b.build()
    }

    fn block_from_dense(
        partitions: &[PartitionSpec],
        h_dense: &DMatrix<f64>,
        ri: usize,
        bi: usize,
        rj: usize,
        bj: usize,
    ) -> DMatrix<f64> {
        let mut offs = Vec::with_capacity(partitions.len());
        let mut o = 0usize;
        for p in partitions {
            offs.push(o);
            o += p.block_count * p.block_dim;
        }

        let (di, dj) = (partitions[ri].block_dim, partitions[rj].block_dim);
        let si = offs[ri] + bi * di;
        let sj = offs[rj] + bj * dj;
        h_dense.view((si, sj), (di, dj)).into_owned()
    }

    #[test]
    fn spd_precision_and_covariance_blocks_match_dense_inverse() {
        // partitions: [(1×2), (2×1)] => n=4
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                block_count: 2,
                block_dim: 1,
            },
        ]);
        let n = 4;

        // SPD H = A^T A + μ I
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let mu = 0.2;
        let h_dense = a.transpose() * &a + mu * DMatrix::<f64>::identity(n, n);

        let mat = build_square_from_dense(parts.clone(), &h_dense);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let r = ldlt.rank();

        let mut gram_schmidt = ldlt.into_min_norm_ldlt();

        println!("rank: {}", r);

        let inv_dense = Cholesky::new(h_dense.clone()).unwrap().inverse();

        println!("A*B={}", h_dense.clone() * inv_dense.clone());

        // a few tiles
        let tiles = [
            ([0usize, 0usize], [0usize, 0usize]), // (0,0) 2×2
            ([1, 1], [0, 0]),                     // (1,1) blk0×blk0 1×1
            ([1, 1], [1, 0]),                     // (1,1) blk1×blk0 1×1
            ([0, 1], [0, 1]),                     // (0,1) 2×1 off-diag
        ];

        for (region_idx, block_idx) in tiles {
            let row_idx = PartitionBlockIndex {
                partition: region_idx[0],
                block: block_idx[0],
            };
            let col_idx = PartitionBlockIndex {
                partition: region_idx[1],
                block: block_idx[1],
            };

            let sij = gram_schmidt.pseudo_inverse_block(row_idx, col_idx);

            let sref = block_from_dense(
                parts.specs(),
                &inv_dense,
                region_idx[0],
                block_idx[0],
                region_idx[1],
                block_idx[1],
            );
            println!("cov {}", sij);
            println!("{}", sref);
            assert_relative_eq!(sij, sref, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn psd_precision_and_covariance_blocks_match_pseudoinverse() {
        // single 3×3 block, rank-2 PSD: H = B^T B
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 3,
        }]);
        let n = 3;

        let b = DMatrix::<f64>::from_row_slice(2, n, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_square_from_dense(parts.clone(), &h_dense);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let mut gram_schmidt = ldlt.into_min_norm_ldlt();

        let pinv = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };

        let sij = gram_schmidt.pseudo_inverse_block(idx00, idx00);
        assert_relative_eq!(sij, pinv, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn off_diagonal_covariance_tile_psd() {
        // two regions: [(1×1), (1×2)] ; n=3 ; rank-2 PSD
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
        ]);
        let b = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_square_from_dense(parts.clone(), &h_dense);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let mut gram_schmidt = ldlt.into_min_norm_ldlt();

        let pinv = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };

        // Tile (region0,blk0) × (region1,blk0)
        let sij = gram_schmidt.pseudo_inverse_block(idx00, idx10);
        let sref = block_from_dense(parts.specs(), &pinv, 0, 0, 1, 0);
        assert_relative_eq!(sij, sref, epsilon = 1e-9, max_relative = 1e-9);
    }

    #[test]
    fn covariance_blocks_are_symmetric() {
        // random SPD system, check S_ij == S_ji^T
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                block_count: 1,
                block_dim: 3,
            },
        ]);
        let n = 5;
        let a = DMatrix::<f64>::from_fn(7, n, |r, c| ((r + 1) * (c + 3)) as f64 / 11.0);
        let h = a.transpose() * &a + 0.3 * DMatrix::<f64>::identity(n, n);

        let mat = build_square_from_dense(parts, &h);
        let ldlt = DenseLdlt::default();
        let ldlt = ldlt.factorize(&mat).unwrap();
        let mut gram_schmidt = ldlt.into_min_norm_ldlt();

        let idx00 = PartitionBlockIndex {
            partition: 0,
            block: 0,
        };
        let idx10 = PartitionBlockIndex {
            partition: 1,
            block: 0,
        };

        let s_01 = gram_schmidt.pseudo_inverse_block(idx00, idx10); // (2×3)
        let s_10 = gram_schmidt.pseudo_inverse_block(idx10, idx00); // (3×2)
        assert_relative_eq!(
            s_01,
            s_10.transpose(),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn full_inverse_matches_dense_spd() {
        // partitions: [(1×2), (2×1)] => n=4
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            },
            PartitionSpec {
                block_count: 2,
                block_dim: 1,
            },
        ]);
        let n = 4;

        // SPD H = Aᵀ A + μ I
        let a = DMatrix::<f64>::from_row_slice(
            3,
            n,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.5, 2.0, 0.0, 0.0, 3.0],
        );
        let mu = 0.2;
        let h_dense = a.transpose() * &a + mu * DMatrix::<f64>::identity(n, n);

        // Factorize & build Gram–Schmidt min-norm system
        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        let gs = ldlt.into_min_norm_ldlt();

        // Our blockwise inverse
        let inv_mat = gs.pseudo_inverse();
        // Convert to a dense matrix for comparison
        let inv_dense_from_blocks = inv_mat;

        // Reference (Cholesky inverse)
        let inv_ref = Cholesky::new(h_dense.clone()).unwrap().inverse();

        assert_relative_eq!(
            inv_dense_from_blocks,
            inv_ref,
            epsilon = 1e-9,
            max_relative = 1e-9
        );
        // sanity: H * H^{-1} ≈ I
        let eye = h_dense * inv_dense_from_blocks;
        assert_relative_eq!(
            eye,
            DMatrix::<f64>::identity(n, n),
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }

    #[test]
    fn full_inverse_matches_pseudoinverse_psd() {
        // single 3×3 block, rank-2 PSD: H = Bᵀ B
        let parts = PartitionSet::new(vec![PartitionSpec {
            block_count: 1,
            block_dim: 3,
        }]);
        let n = 3;

        let b = DMatrix::<f64>::from_row_slice(2, n, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let h_dense = b.transpose() * &b;

        let mat = build_square_from_dense(parts, &h_dense);
        let ldlt = DenseLdlt::default().factorize(&mat).unwrap();
        let gs = ldlt.into_min_norm_ldlt();

        let inv_mat = gs.pseudo_inverse();
        let inv_dense_from_blocks = inv_mat;

        let pinv_ref = pseudo_inverse_with_tol(h_dense.as_view(), LDLT_TOL);

        assert_relative_eq!(
            inv_dense_from_blocks,
            pinv_ref,
            epsilon = 1e-9,
            max_relative = 1e-9
        );
        // Moore–Penrose consistency: H * H^+ * H ≈ H
        let check = &h_dense * &inv_dense_from_blocks * &h_dense;
        assert_relative_eq!(check, h_dense, epsilon = 1e-9, max_relative = 1e-9);
    }
}
