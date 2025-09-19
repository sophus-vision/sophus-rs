///s
pub mod block_sparse_min_norm_ldlt;
/// d
pub mod dense_min_norm_ldlt;
///s
pub mod sparse_min_norm_ldlt;

use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    IsMinNormFactor,
    kernel::{
        diag_matsolve_inplaced,
        diag_right_matsolve_inplace,
    },
    ldlt::{
        ldlt_decompose_inplace,
        ldlt_matsolve_inplace,
        ldlt_right_matsolve_inplace,
    },
    matrix::{
        BlockRange,
        PartitionBlockIndex,
    },
};

/// Min-norm LDLᵀ to calculate the matrix (pseudo) inverse.
///
/// If `A` is full-rank, then `A x = b` has a unique solution `x`. Furthermore,
/// the inverse of `A` can be obtained by solving `A X = I`.
///
/// If `A` is rank deficient, then `A x = b` has multiple solutions and  `A`  has
/// no matrix inverse. However, one can calculate the pseudo-inverse or Moore–Penrose inverse of A
/// instead. It is the smallest `|X|` such that `A X = I`, hence the min-norm solution of
/// `A X = I`.
///
/// Another way of looking at it that the pseudo-inverse A* can be defined using the limit (ε -> 0):
///
///  A* := limᵋ (AᵀA + εI) Aᵀ
///
/// ...
///
/// Let `E = L[:,J]`. Then the gram-schmidt matrix is `G = Eᵀ E`. Now we perform
/// a LDLᵀ  of G:   Lᴳ, and dᴳ.
///
/// Solve: Lᴳ Z = * I
/// solve: diag(dᴳ) Y = * Z
/// solve: (Lᴳ)ᵀ X = Y

#[derive(Clone, Debug)]
pub struct MinNormLdlt<Backend: IsMinNormLdltBackend> {
    backend: Backend,
    gram_mat_l: DMatrix<f64>,
    gram_diag: DVector<f64>,
    mat_e: DMatrix<f64>,
}

impl<F: IsMinNormLdltBackend> MinNormLdlt<F> {
    /// inverse
    pub fn new(fact: F::LdltFactor) -> Self {
        let f = F::new(fact);

        let n = f.scalar_dim();
        let r = f.rank();

        // --- Build E = L[:,J] (n × r)
        let mut e = DMatrix::<f64>::zeros(n, r);
        for (c, &t) in f.positive_pivot_idx().iter().enumerate() {
            f.column_of_mat_e(t, e.column_mut(c).as_mut_slice());
        }
        // --- G = Eᵀ E (r × r), then LDLᵀ(G)
        let mut g = e.transpose() * &e;
        let mut g_diag = DVector::<f64>::zeros(r);
        ldlt_decompose_inplace(g.as_view_mut(), g_diag.as_view_mut(), f.tol_rel())
            .expect("G must be SPD on the positive-pivot subspace");
        Self {
            backend: f,
            gram_diag: g_diag,
            gram_mat_l: g,
            mat_e: e,
        }
    }
}

impl<F: IsMinNormLdltBackend> IsMinNormFactor for MinNormLdlt<F> {
    fn pseudo_inverse(&self) -> DMatrix<f64> {
        let n = self.backend.scalar_dim();
        let r = self.backend.rank();

        if n == 0 || r == 0 {
            return DMatrix::<f64>::zeros(n, n);
        }

        // Fast SPD path if the backend can do it:
        if let Some(spd_inv) = self.backend.try_inverse() {
            return spd_inv;
        }
        // If not provided, the generic E/G path also yields the exact inverse when r == n:
        // take J = {0..n-1}, E = L, G = Lᵀ L => E G^{-1} D^{-1} G^{-1} Eᵀ = L L^{-1} L^{-T} D^{-1}
        // L^{-1} L^{-T} Lᵀ = L^{-T} D^{-1} L^{-1}.

        // --- A = E * G^{-1} * D_J^{-1} * G^{-1}  (n × r)
        let mut a = self.mat_e.clone();
        ldlt_right_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            a.as_view_mut(),
        );
        diag_right_matsolve_inplace(
            self.backend.positive_pivot_values().as_view(),
            &mut a.as_view_mut(),
        );
        ldlt_right_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            a.as_view_mut(),
        );

        // --- S = A * Eᵀ  (n × n)
        let mut out = DMatrix::<f64>::zeros(n, n);
        a.mul_to(&self.mat_e.transpose(), &mut out);
        out
    }

    fn pseudo_inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        let r = self.backend.rank();

        let row_range = self.backend.block_range(row_idx);
        let col_range = self.backend.block_range(col_idx);
        let mut out = DMatrix::<f64>::zeros(row_range.block_dim, col_range.block_dim);

        if r == 0 || row_range.block_dim == 0 || col_range.block_dim == 0 {
            return out;
        }

        // If the backend provides a fast SPD inverse, slice the block out of it.
        if let Some(cols) = self.backend.try_column_of_inverse(&col_range) {
            // cols is n × dj, extract the wanted row-block.
            return cols
                .view(
                    (row_range.start_idx, 0),
                    (row_range.block_dim, col_range.block_dim),
                )
                .into_owned();
        }

        // --- Form the small tiles:
        // A = L_{iJ} = E[rows_i, :]          (di × r)
        // B = L_{jJ}ᵀ = E[rows_j, :]ᵀ        (r  × dj)
        let a = self
            .mat_e
            .view((row_range.start_idx, 0), (row_range.block_dim, r))
            .into_owned();
        let mut b = self
            .mat_e
            .view((col_range.start_idx, 0), (col_range.block_dim, r))
            .transpose()
            .into_owned();

        // --- Apply M = G^{-1} D_J^{-1} G^{-1} to B on the left:
        // b <- G^{-1} b
        ldlt_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            b.as_view_mut(),
        );
        // b <- D_J^{-1} b   (row-wise scaling by positive pivots)
        diag_matsolve_inplaced(
            self.backend.positive_pivot_values().as_view(),
            &mut b.as_view_mut(),
        );
        // b <- G^{-1} b
        ldlt_matsolve_inplace(
            self.gram_mat_l.as_view(),
            self.gram_diag.as_view(),
            b.as_view_mut(),
        );

        // --- Multiply out = A * b
        a.mul_to(&b, &mut out);

        out
    }
}

/// What the min-norm algorithm needs from an LDLᵀ backend.
pub trait IsMinNormLdltBackend {
    /// The underlying LDLt factorization of matrix A.
    type LdltFactor;

    /// Create a new min-norm LDLt.
    fn new(ldlt: Self::LdltFactor) -> Self;

    /// The scalar dimension of the square matrices A = L D Lᵀ.
    fn scalar_dim(&self) -> usize;

    /// The rank of the matrix A = L D Lᵀ.
    fn rank(&self) -> usize {
        debug_assert_eq!(
            self.positive_pivot_idx().len(),
            self.positive_pivot_values().len()
        );

        self.positive_pivot_idx().len()
    }

    /// Relative tolerance, used to decompose the gram matrix G.
    fn tol_rel(&self) -> f64;

    /// Indices of positive pivots (len = rank).
    fn positive_pivot_idx(&self) -> &[usize];

    /// Vector of positive values (len = rank).
    fn positive_pivot_values(&self) -> &DVector<f64>;

    /// Emit the scalar column `L[:, j]` into `out` (length n), with the **unit-diagonal
    /// convention**: `out[j]` must be 1.0; entries above t are 0.0; for rows > t, the stored
    /// strict-lower values.
    fn column_of_mat_e(&self, col_j: usize, out: &mut [f64]);

    /// Tries to calculate the inverse of `A = L D Lᵀ`. If `A` is rank-deficient, then None is
    /// returned.
    fn try_inverse(&self) -> Option<DMatrix<f64>>;

    /// Tries to calculate a block-column of the inverse of `A = L D Lᵀ`. If `A` is rank-deficient,
    /// then None is returned.
    fn try_column_of_inverse(&self, col_range: &BlockRange) -> Option<nalgebra::DMatrix<f64>>;

    /// Return the block range for a given partition of A.
    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange;
}
