use crate::sparse::CscStruct;

const INVALID: usize = usize::MAX;

pub(crate) struct EliminationTree {
    pub(crate) parent: Vec<usize>,
    pub(crate) w: Vec<usize>,
    pub(crate) stack: Vec<usize>,
}

impl EliminationTree {
    /// Symbolic reach on the **upper** structure (Aᵗ).
    /// Returns top-of-stack index into `stack`, so `stack[top..n]` are the columns
    /// k (< j) in topological order. Pre-marks `j` to avoid returning it.
    pub(crate) fn reach(&mut self, upper_mat_a: &CscStruct, j: usize) -> usize {
        let mark = j + 1;
        let mut top = upper_mat_a.n;

        // Block j from appearing in the reach
        self.w[j] = mark;

        for p in upper_mat_a.col_ptr[j]..upper_mat_a.col_ptr[j + 1] {
            let mut i = upper_mat_a.row_ind[p];
            if i >= j {
                continue;
            } // strictly upper start points
            while i != INVALID && self.w[i] != mark {
                self.stack[top - 1] = i;
                top -= 1;
                self.w[i] = mark;
                i = self.parent[i];
            }
        }
        top
    }
}

/// Elimination tree using the **upper** structure (from Aᵗ).
/// parent[v] = parent column of v, or INVALID if root.
/// Davis, "Direct Methods...", Alg. 4.1 (upper form).
pub(crate) fn elimination_tree_upper(at_upper: &CscStruct) -> EliminationTree {
    let n = at_upper.n;
    let mat_a_p = &at_upper.col_ptr;
    let mat_a_i = &at_upper.row_ind;

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
        w: vec![0; n],
        stack: vec![0; n],
    }
}
