use nonmax::NonMaxUsize;

use crate::sparse::CscPattern;

struct EliminationTreeWorkspace {
    stamp: Vec<usize>,
    reach_buf: Vec<usize>,
    tick: usize,
}

/// Elimination tree - used by sparse LDLᵀ decomposition
pub struct EliminationTree {
    upper_pattern: CscPattern,
    parent_array: Vec<Option<NonMaxUsize>>,
    workspace: EliminationTreeWorkspace,
}

impl EliminationTree {
    /// Creates new from upper structure of the symmetric matrix A.
    pub fn new(upper_pattern: CscPattern) -> Self {
        let n = upper_pattern.row_count();

        let mut parent: Vec<Option<NonMaxUsize>> = vec![None; n];
        let mut ancestor: Vec<Option<NonMaxUsize>> = vec![None; n];

        for j in 0..n {
            for q in upper_pattern.col_ptr()[j]..upper_pattern.col_ptr()[j + 1] {
                let i0 = upper_pattern.row_idx()[q];
                if i0 >= j {
                    continue; // strictly upper: i < j
                }
                let mut i = i0;
                // Path-compress along 'ancestor' towards root.

                // TODO: explain better
                while i != j {
                    let next = ancestor[i];
                    ancestor[i] = NonMaxUsize::new(j);
                    if next.is_none() {
                        parent[i] = NonMaxUsize::new(j);
                        break;
                    }
                    i = next.unwrap().get();
                }
            }
        }

        EliminationTree {
            upper_pattern,
            parent_array: parent,
            workspace: EliminationTreeWorkspace {
                stamp: vec![0; n],
                reach_buf: vec![0; n],
                tick: 0,
            },
        }
    }

    /// Symbolic reach on the upper structure of the symmetric matrix A.
    ///
    /// Return list of columns i with  i<j which can be reached by column j.
    ///
    ///  - These are columns i which are adjacent to column j (i.e. non-zero at L(i,j)).
    ///  - Columns on the path (via elimination tree) between adjacent columns and column j.
    pub fn reach(&mut self, j: usize) -> &[usize] {
        self.workspace.tick = self.workspace.tick.wrapping_add(1);
        assert_ne!(
            self.workspace.tick, 0,
            "fn called more than usize::MAX times?"
        );

        let mark = self.workspace.tick;

        let n = self.upper_pattern.row_count();
        let mut top = n;

        // Block j from appearing in the reach
        self.workspace.stamp[j] = mark;

        let col_ptr = self.upper_pattern.col_ptr();
        let row_idx = self.upper_pattern.row_idx();

        for q in col_ptr[j]..col_ptr[j + 1] {
            let mut i = row_idx[q];
            if i >= j {
                continue;
            } // strictly upper neighbors
            loop {
                if self.workspace.stamp[i] == mark {
                    break;
                }
                self.workspace.reach_buf[top - 1] = i;
                top -= 1;
                self.workspace.stamp[i] = mark;

                match self.parent_array[i] {
                    Some(pi) => i = pi.get(),
                    None => break, // reached a root
                }
            }
        }
        &self.workspace.reach_buf[top..n]
    }

    /// Access the parent array. Roots are `None`.
    #[inline]
    pub fn parent_array(&self) -> &[Option<NonMaxUsize>] {
        &self.parent_array
    }
}

#[cfg(test)]
mod test {

    use sophus_autodiff::linalg::{
        IsMatrix,
        MatF64,
    };

    use super::*;
    use crate::{
        assert_le,
        sparse::{
            CscMatrix,
            TripletMatrix,
        },
    };

    // Helper: build an N×N CSC pattern for upper(A) from a list of strictly‑upper edges (i, j)
    // with i < j. We go via TripletMatrix → CscMatrix to get canonical CSC (sorted rows per
    // column).
    fn upper_from_edges(n: usize, edges: &[(usize, usize)]) -> CscMatrix {
        for &(i, j) in edges {
            assert_le!(i, j,);
            assert_le!(i, n);
            assert_le!(j, n);
        }
        let triplets: Vec<(usize, usize, f64)> = edges.iter().map(|&(i, j)| (i, j, 1.0)).collect();
        CscMatrix::from_triplets(&TripletMatrix::new(triplets, n, n))
    }

    fn parent_of_col_desc(etree: &EliminationTree, j: usize) -> String {
        let parent_of_col_j = etree.parent_array()[j];
        if parent_of_col_j.is_none() {
            return format!("Column {j} is root.").to_owned();
        }
        format!(
            "Parent of column {j} is column {}.",
            parent_of_col_j.unwrap()
        )
    }

    fn reach_of_col_desc(etree: &mut EliminationTree, j: usize) -> String {
        let reach_of_col_j = etree.reach(j);
        if reach_of_col_j.is_empty() {
            return format!("Column {j} reaches no earlier columns.");
        }
        format!(
            "Column {j} reaches these earlier columns \
            (directly or via the elimination tree): {reach_of_col_j:?}."
        )
    }

    #[test]
    fn etree_linear_chain_parents() {
        let n = 5;
        let upper_csc = upper_from_edges(n, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(
            upper_csc.to_dense(),
            MatF64::<5, 5>::from_array2([
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])
        );
        let mut etree = EliminationTree::new(upper_csc.pattern().clone());

        assert_eq!(
            parent_of_col_desc(&etree, 0),
            "Parent of column 0 is column 1.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 1),
            "Parent of column 1 is column 2.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 2),
            "Parent of column 2 is column 3.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 3),
            "Parent of column 3 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 4),
            "Column 4 is root.".to_owned()
        );

        assert_eq!(
            reach_of_col_desc(&mut etree, 0),
            "Column 0 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 1),
            "Column 1 reaches these earlier columns (directly or via the elimination tree): [0]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 2),
            "Column 2 reaches these earlier columns (directly or via the elimination tree): [1]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 3),
            "Column 3 reaches these earlier columns (directly or via the elimination tree): [2]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 4),
            "Column 4 reaches these earlier columns (directly or via the elimination tree): [3]."
                .to_owned()
        );
    }

    #[test]
    fn reach_chain_with_far_neighbor_topological_order() {
        let n = 5;
        let upper_csc = upper_from_edges(n, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
        assert_eq!(
            upper_csc.to_dense(),
            MatF64::<5, 5>::from_array2([
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])
        );
        let mut etree = EliminationTree::new(upper_csc.pattern().clone());

        assert_eq!(
            parent_of_col_desc(&etree, 0),
            "Parent of column 0 is column 1.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 1),
            "Parent of column 1 is column 2.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 2),
            "Parent of column 2 is column 3.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 3),
            "Parent of column 3 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 4),
            "Column 4 is root.".to_owned()
        );

        assert_eq!(
            reach_of_col_desc(&mut etree, 0),
            "Column 0 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 1),
            "Column 1 reaches these earlier columns (directly or via the elimination tree): [0]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 2),
            "Column 2 reaches these earlier columns (directly or via the elimination tree): [1]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 3),
            "Column 3 reaches these earlier columns (directly or via the elimination tree): [2]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 4),
            "Column 4 reaches these earlier columns (directly or via the elimination tree): [3, 2, 1, 0]."
                .to_owned()
        );
    }

    #[test]
    fn etree_star_from_single_column() {
        let n = 5;
        let upper_csc = upper_from_edges(n, &[(0, 4), (1, 4), (2, 4), (3, 4)]);
        assert_eq!(
            upper_csc.to_dense(),
            MatF64::<5, 5>::from_array2([
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])
        );
        let mut etree = EliminationTree::new(upper_csc.pattern().clone());

        assert_eq!(
            parent_of_col_desc(&etree, 0),
            "Parent of column 0 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 1),
            "Parent of column 1 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 2),
            "Parent of column 2 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 3),
            "Parent of column 3 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 4),
            "Column 4 is root.".to_owned()
        );

        assert_eq!(
            reach_of_col_desc(&mut etree, 0),
            "Column 0 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 1),
            "Column 1 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 2),
            "Column 2 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 3),
            "Column 3 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 4),
            "Column 4 reaches these earlier columns (directly or via the elimination tree): [3, 2, 1, 0]."
                .to_owned()
        );
    }

    #[test]
    fn etree_two_branches_then_join() {
        let n = 5;
        let upper_csc = upper_from_edges(n, &[(0, 1), (2, 3), (1, 4), (3, 4)]);
        assert_eq!(
            upper_csc.to_dense(),
            MatF64::<5, 5>::from_array2([
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])
        );
        let mut etree = EliminationTree::new(upper_csc.pattern().clone());

        assert_eq!(
            parent_of_col_desc(&etree, 0),
            "Parent of column 0 is column 1.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 1),
            "Parent of column 1 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 2),
            "Parent of column 2 is column 3.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 3),
            "Parent of column 3 is column 4.".to_owned()
        );
        assert_eq!(
            parent_of_col_desc(&etree, 4),
            "Column 4 is root.".to_owned()
        );

        assert_eq!(
            reach_of_col_desc(&mut etree, 0),
            "Column 0 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 1),
            "Column 1 reaches these earlier columns (directly or via the elimination tree): [0]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 2),
            "Column 2 reaches no earlier columns.".to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 3),
            "Column 3 reaches these earlier columns (directly or via the elimination tree): [2]."
                .to_owned()
        );
        assert_eq!(
            reach_of_col_desc(&mut etree, 4),
            "Column 4 reaches these earlier columns (directly or via the elimination tree): [3, 1]."
                .to_owned()
        );
    }

    #[test]
    fn ignores_lower_and_diagonal_entries() {
        let n = 4;

        // Clean upper edges (define the true etree)
        let strictly_upper = upper_from_edges(n, &[(0, 1), (1, 2)]);
        assert_eq!(
            strictly_upper.to_dense(),
            MatF64::<4, 4>::from_array2([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ])
        );

        //  Same but we pollute with diagonal/lower entries that should be ignored by the algorithm.
        let noisy_triplets = TripletMatrix::new(
            vec![
                (0, 1, 1.0), // upper
                (1, 2, 1.0), // upper
                // noise:
                (1, 1, 1.0), // diag
                (3, 1, 1.0), // lower
                (2, 2, 1.0), // diag
                (3, 3, 1.0), // diag
                (3, 2, 1.0), // lower
            ],
            n,
            n,
        );
        let noisy = CscMatrix::from_triplets(&noisy_triplets);
        assert_eq!(
            noisy.to_dense(),
            MatF64::<4, 4>::from_array2([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
            ])
        );
        let mut etree_upper = EliminationTree::new(strictly_upper.pattern().clone());
        let mut etree_noisy = EliminationTree::new(noisy.pattern().clone());

        assert_eq!(etree_upper.parent_array(), etree_noisy.parent_array());

        for j in 0..4 {
            assert_eq!(etree_upper.reach(j), etree_noisy.reach(j));
        }
    }
}
