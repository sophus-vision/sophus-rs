use sophus_autodiff::linalg::MatF64;
use sophus_solver::{
    matrix::{
        ColumnCompressedMatrix,
        PartitionSpec,
        SparseSymmetricMatrixBuilder,
    },
    positive_semidefinite::{
        EliminationTree,
        IsLdltTracer,
        LdltIndices,
        LdltWorkspace,
        SparseLFactorBuilder,
        SparseLdlt,
    },
    prelude::*,
};

/// LDLᵀ ascii tracer.
pub struct AsciiTracer {}

impl IsLdltTracer<LdltWorkspace> for AsciiTracer {
    #[inline]
    fn after_etree(&mut self, etree: &EliminationTree) {
        let parent = etree.parent_array();
        let n = parent.len();

        let mut child_arrays: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut root_array: Vec<usize> = Vec::new();

        // Build children arrays.
        for (parent_idx, parent) in parent.iter().enumerate() {
            match parent {
                Some(parent) => child_arrays[parent.get()].push(parent_idx),
                None => root_array.push(parent_idx),
            }
        }

        // Sort arrays.
        for children in &mut child_arrays {
            children.sort_unstable();
        }
        root_array.sort_unstable();

        fn record_node(
            node: usize,
            child_arrays: &[Vec<usize>],
            prefix: &str,
            last: bool,
            out: &mut String,
        ) {
            out.push_str(prefix);
            out.push_str(if last { "`-- " } else { "|-- " });
            out.push_str(&format!("{node}\n"));

            let next_prefix = if last {
                format!("{prefix}    ")
            } else {
                format!("{prefix}|   ")
            };

            let children = &child_arrays[node];
            for (child_idx, &child) in children.iter().enumerate() {
                record_node(
                    child,
                    child_arrays,
                    &next_prefix,
                    child_idx + 1 == children.len(),
                    out,
                );
            }
        }

        // Emit each root and its subtree
        let mut out = String::new();
        for (ri, &r) in root_array.iter().enumerate() {
            // If multiple roots, label them by index; otherwise just print the root number
            if root_array.len() > 1 {
                out.push_str(&format!("root[{ri}]: {r}\n"));
            } else {
                out.push_str(&format!("{r}\n"));
            }
            let children = &child_arrays[r];
            for (child_idx, &child) in children.iter().enumerate() {
                record_node(
                    child,
                    &child_arrays,
                    "",
                    child_idx + 1 == children.len(),
                    &mut out,
                );
            }
        }
        println!("{out}");
    }
    #[inline]
    fn after_load_column_and_reach(&mut self, j: usize, reach: &[usize], _ws: &LdltWorkspace) {
        println!("Column {j} is connected to those columns on the left: {reach:?}");

        println!("y={:?}", &_ws.c);
        println!("stamped={:?}", _ws.was_row_touched);
        println!("touch={:?}", _ws.touched_rows);
    }
    #[inline]
    fn after_update(&mut self, indices: LdltIndices, d_k: f64, l_ik: f64, l_jk: f64, c_i: f64) {
        let (i, j, k) = (indices.row_i, indices.col_j, indices.col_k);
        println!(
            "Subtract  L({j},{k}) * d[{k}] * L({i},{k}) = {:.3} * {:.3} * {:.3} = {}  from y[{i}]",
            l_jk,
            d_k,
            l_ik,
            l_jk * d_k * l_ik
        );

        println!("c[{}] = {:?}", i, c_i);
    }
    #[inline]
    fn after_append_and_sort(&mut self, _j: usize, _l_storage: &SparseLFactorBuilder, d: &[f64]) {
        //println!("L={:1.3}", l_storage.compress().to_dense());
        println!("d={d:?}");
    }
}

fn main() {
    let partitions = vec![PartitionSpec {
        block_count: 2,
        block_dimension: 3,
    }];
    let mut builder = SparseSymmetricMatrixBuilder::zero(&partitions);

    let a00 = MatF64::<3, 3>::from_array2([
        [3.3, 0.0, 0.0], //
        [0.0, 3.2, 0.0],
        [1.1, 1.2, 3.1],
    ]);
    let a10 = MatF64::<3, 3>::from_array2([
        [0.0, 0.0, 0.0], //
        [0.0, 0.0, 0.0],
        [0.0, 1.3, 0.0],
    ]);
    let a11 = MatF64::<3, 3>::from_array2([
        [2.3, 0.0, 0.0], //
        [0.9, 2.2, 0.0],
        [0.0, 1.0, 2.1],
    ]);

    builder.add_lower_block(&[0, 0], [0, 0], &a00.as_view());
    builder.add_lower_block(&[0, 0], [1, 0], &a10.as_view());
    builder.add_lower_block(&[0, 0], [1, 1], &a11.as_view());

    let lower_mat_a = ColumnCompressedMatrix::from_triplets(&builder.build());
    let dense_lower_mat_a = lower_mat_a.triangular_to_dense_symmetric();

    print!("lower(A) = \n{dense_lower_mat_a:.3}");

    let mut tracer = AsciiTracer {};
    let solver = SparseLdlt::default();
    solver.factorize(&lower_mat_a, &mut tracer).unwrap();
}
