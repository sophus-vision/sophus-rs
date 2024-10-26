use crate::cost::IsCost;
use crate::ldlt::SparseLdlt;
use crate::ldlt::SymmetricTripletMatrix;
use crate::variables::VarKind;
use crate::variables::VarPool;

extern crate alloc;

/// Normal equation
pub struct SparseNormalEquation {
    sparse_hessian: SymmetricTripletMatrix,
    neg_gradient: nalgebra::DVector<f64>,
}

impl SparseNormalEquation {
    fn from_families_and_cost(
        variables: &VarPool,
        costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsCost>>,
        nu: f64,
    ) -> SparseNormalEquation {
        assert!(variables.num_of_kind(VarKind::Marginalized) == 0);
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        // Note let's first focus on these special cases, before attempting a
        // general version covering all cases holistically. Also, it might not be trivial
        // to implement VarKind::Marginalized > 1.
        //  -  Example, the the arrow-head sparsity uses a recursive application of the Schur-Complement.
        let num_var_params = variables.num_free_params();
        let mut upper_hessian_triplet = alloc::vec::Vec::new();
        let mut neg_grad = nalgebra::DVector::<f64>::zeros(num_var_params);

        for cost in costs.iter() {
            cost.populate_normal_equation(variables, nu, &mut upper_hessian_triplet, &mut neg_grad);
        }

        Self {
            sparse_hessian: SymmetricTripletMatrix {
                upper_triplets: upper_hessian_triplet,
                size: num_var_params,
            },

            neg_gradient: neg_grad,
        }
    }

    fn solve(&mut self) -> nalgebra::DVector<f64> {
        // TODO: perform permutation to reduce fill-in and symbolic factorization only once
        SparseLdlt::from_triplets(&self.sparse_hessian).solve(&self.neg_gradient)
    }
}

/// Solve the normal equation
pub fn solve(
    variables: &VarPool,
    costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsCost>>,
    nu: f64,
) -> VarPool {
    assert!(variables.num_of_kind(VarKind::Marginalized) <= 1);
    assert!(variables.num_of_kind(VarKind::Free) >= 1);

    if variables.num_of_kind(VarKind::Marginalized) == 0 {
        let mut sne = SparseNormalEquation::from_families_and_cost(variables, costs, nu);
        sne.solve();
        let delta = sne.solve();
        variables.update(delta)
    } else {
        todo!()
    }
}
