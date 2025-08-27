use crate::sparse::CscPattern;

const INVALID: usize = usize::MAX;

pub(crate) struct EliminationTree {
    pub(crate) parent: Vec<usize>,
    pub(crate) stamp: Vec<usize>,
    pub(crate) reach_buf: Vec<usize>,
}

impl EliminationTree {
    /// Symbolic reach on the **upper** structure (Aᵗ).
    /// Returns top-of-stack index into `stack`, so `stack[top..n]` are the columns
    /// k (< j) in topological order. Pre-marks `j` to avoid returning it.
    pub(crate) fn reach(&mut self, upper_mat_a: &CscPattern, j: usize) -> usize {
        let mark = j + 1;
        let mut top = upper_mat_a.row_count();

        // Block j from appearing in the reach
        self.stamp[j] = mark;

        for p in upper_mat_a.col_ptr()[j]..upper_mat_a.col_ptr()[j + 1] {
            let mut i = upper_mat_a.row_idx()[p];
            if i >= j {
                continue;
            } // strictly upper start points
            while i != INVALID && self.stamp[i] != mark {
                self.reach_buf[top - 1] = i;
                top -= 1;
                self.stamp[i] = mark;
                i = self.parent[i];
            }
        }
        top
    }
}

/// Elimination tree using the **upper** structure (from Aᵗ).
/// parent[v] = parent column of v, or INVALID if root.
/// Davis, "Direct Methods...", Alg. 4.1 (upper form).
pub(crate) fn elimination_tree_upper(at_upper: &CscPattern) -> EliminationTree {
    let n = at_upper.row_count();
    let mat_a_p = &at_upper.col_ptr();
    let mat_a_i = &at_upper.row_idx();

    let mut parent = vec![INVALID; n];
    let mut ancestor = vec![INVALID; n];

    for j in 0..n {
        for p in mat_a_p[j]..mat_a_p[j + 1] {
            let i0 = mat_a_i[p];
            if i0 >= j {
                continue;
            } // strictly upper: i < j
            let mut i = i0;
            while i != INVALID && i != j {
                let next = ancestor[i];
                ancestor[i] = j;
                if next == INVALID {
                    parent[i] = j;
                    break;
                }
                i = next;
            }
        }
    }
    EliminationTree {
        parent,
        stamp: vec![0; n],
        reach_buf: vec![0; n],
    }
}
