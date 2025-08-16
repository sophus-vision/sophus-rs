use crate::sparse::CscStruct;

const INVALID: usize = usize::MAX;

/// Elimination tree using the **upper** structure (from Aᵗ).
/// parent[v] = parent column of v, or INVALID if root.
/// Davis, "Direct Methods...", Alg. 4.1 (upper form).
pub(crate) fn elimination_tree_upper(at_upper: &CscStruct) -> Vec<usize> {
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
    parent
}

/// Symbolic reach on the **upper** structure (Aᵗ).
/// Returns top-of-stack index into `stack`, so `stack[top..n]` are the columns
/// k (< j) in topological order. Pre-marks `j` to avoid returning it.
pub(crate) fn ereach_upper(
    at_upper: &CscStruct,
    j: usize,
    parent: &[usize],
    w: &mut [usize],     // stamp marks
    stack: &mut [usize], // length n
) -> usize {
    let n = at_upper.n;
    let mat_a_p = &at_upper.col_ptr;
    let mat_a_i = &at_upper.row_ind;

    let mark = j + 1;
    let mut top = n;

    // Block j from appearing in the reach
    w[j] = mark;

    for p in mat_a_p[j]..mat_a_p[j + 1] {
        let mut i = mat_a_i[p];
        if i >= j {
            continue;
        } // strictly upper start points
        while i != INVALID && w[i] != mark {
            stack[top - 1] = i;
            top -= 1;
            w[i] = mark;
            i = parent[i];
        }
    }
    top
}
