use core::fmt::Debug;

use dyn_clone::DynClone;
use sophus_autodiff::manifold::IsVariable;

use super::VarKind;

extern crate alloc;

/// A generic family of variables
///
/// A list of variables of the same nature (e.g., a list of 3D points, a list of 2D isometries,
/// etc.)
#[derive(Debug, Clone)]
pub struct VarFamily<Var: IsVariable> {
    kind: VarKind,
    /// members of the family
    pub members: alloc::vec::Vec<Var>,
    constant_members: alloc::collections::BTreeMap<usize, ()>,
    scalar_indices: alloc::vec::Vec<i64>,
    block_indices: alloc::vec::Vec<i64>,
}

impl<Var: IsVariable> VarFamily<Var> {
    /// create a new variable family a variable kind (free, conditioned, ...) and a list of members
    pub fn new(kind: VarKind, members: alloc::vec::Vec<Var>) -> Self {
        Self::new_with_const_ids(kind, members, alloc::collections::BTreeMap::new())
    }

    /// Create a new variable family a variable kind (free, conditioned, ...), a list of members and
    /// a list of constant members
    ///
    /// The constant members are not updated during optimization (no matter the variable kind)
    pub fn new_with_const_ids(
        kind: VarKind,
        members: alloc::vec::Vec<Var>,
        constant_members: alloc::collections::BTreeMap<usize, ()>,
    ) -> Self {
        VarFamily {
            kind,
            members,
            constant_members,
            scalar_indices: alloc::vec![],
            block_indices: alloc::vec![],
        }
    }

    fn c(&self) -> VarKind {
        self.kind
    }
}

/// A family of variables
pub trait IsVarFamily: as_any::AsAny + Debug + DynClone {
    /// update variables in-place (called during optimization)
    fn update(&mut self, delta: nalgebra::DVectorView<f64>);

    /// update the ith variable in-place (called during optimization)
    fn update_i(&mut self, i: usize, delta: nalgebra::DVector<f64>);

    /// total number of free scalars
    fn num_free_vars(&self) -> usize;

    /// total number of free scalars (#free variables * #DOF)
    fn num_free_scalars(&self) -> usize;

    /// start indices in linear system of equations
    fn calc_start_indices(&mut self, scalar_offset: &mut usize);

    /// total number of marginalized scalars (#marginalized variables * #DOF)
    fn num_marg_scalars(&self) -> usize;

    /// start indices in linear system of equations
    fn get_scalar_start_indices(&self) -> &alloc::vec::Vec<i64>;

    /// start indices in linear system of equations
    fn get_block_start_indices(&self) -> &alloc::vec::Vec<i64>;

    /// number of members in the family
    fn len(&self) -> usize;

    /// is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns 0 if variable is conditioned, DOF otherwise
    fn free_or_marg_dof(&self) -> usize;

    /// variable kind (free, conditioned, ...)
    fn get_var_kind(&self) -> VarKind;

    /// concrete type name
    fn concrete_type_name(&self) -> &'static str;
}

impl<Var: IsVariable + 'static> IsVarFamily for VarFamily<Var> {
    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let dof = self.free_or_marg_dof();

        assert_eq!(self.scalar_indices.len(), self.members.len());

        for i in 0..self.scalar_indices.len() {
            let start_idx = self.scalar_indices[i];
            if start_idx == -1 {
                continue;
            }
            let start_idx = start_idx as usize;
            self.members[i].update(delta.rows(start_idx, dof));
        }
    }

    fn len(&self) -> usize {
        self.members.len()
    }

    fn num_free_vars(&self) -> usize {
        match self.get_var_kind() {
            VarKind::Free => self.members.len() - self.constant_members.len(),
            VarKind::Conditioned => 0,
            VarKind::Marginalized => 0,
        }
    }

    fn num_free_scalars(&self) -> usize {
        match self.get_var_kind() {
            VarKind::Free => (self.members.len() - self.constant_members.len()) * Var::DOF,
            VarKind::Conditioned => 0,
            VarKind::Marginalized => 0,
        }
    }

    fn num_marg_scalars(&self) -> usize {
        match self.get_var_kind() {
            VarKind::Free => 0,
            VarKind::Conditioned => 0,
            VarKind::Marginalized => (self.members.len() - self.constant_members.len()) * Var::DOF,
        }
    }

    // returns -1 if variable is not free
    fn calc_start_indices(&mut self, inout_scalar_offset: &mut usize) {
        assert_eq!(
            self.scalar_indices.len(),
            0,
            "This function must only called once"
        );
        assert_eq!(
            self.block_indices.len(),
            0,
            "This function must only called once"
        );

        match self.get_var_kind() {
            VarKind::Free => {
                let mut scalar_indices = alloc::vec![];
                let mut block_indices = alloc::vec![];

                let mut scalar_idx: usize = *inout_scalar_offset;
                let mut block_idx: usize = 0;

                for i in 0..self.members.len() {
                    if self.constant_members.contains_key(&i) {
                        scalar_indices.push(-1);
                        block_indices.push(-1);
                    } else {
                        scalar_indices.push(scalar_idx as i64);
                        block_indices.push(block_idx as i64);
                        scalar_idx += Var::DOF;
                        block_idx += 1;
                    }
                }
                *inout_scalar_offset = scalar_idx;

                assert_eq!(scalar_indices.len(), self.members.len());
                assert_eq!(block_indices.len(), self.members.len());

                self.scalar_indices = scalar_indices;
                self.block_indices = block_indices;
            }
            VarKind::Conditioned => {
                let mut scalar_indices = alloc::vec![];
                let mut block_indices = alloc::vec![];

                for _i in 0..self.members.len() {
                    scalar_indices.push(-1);
                    block_indices.push(-1);
                }
                assert_eq!(scalar_indices.len(), self.members.len());
                assert_eq!(block_indices.len(), self.members.len());

                self.scalar_indices = scalar_indices;
                self.block_indices = block_indices;
            }
            VarKind::Marginalized => {
                let mut scalar_indices = alloc::vec![];
                let mut block_indices = alloc::vec![];

                for _i in 0..self.members.len() {
                    scalar_indices.push(-2);
                    block_indices.push(-2);
                }
                assert_eq!(scalar_indices.len(), self.members.len());
                assert_eq!(block_indices.len(), self.members.len());

                self.scalar_indices = scalar_indices;
                self.block_indices = block_indices;
            }
        }
    }

    fn free_or_marg_dof(&self) -> usize {
        match self.get_var_kind() {
            VarKind::Free => Var::DOF,
            VarKind::Conditioned => 0,
            VarKind::Marginalized => Var::DOF,
        }
    }

    fn get_var_kind(&self) -> VarKind {
        self.c()
    }

    fn get_scalar_start_indices(&self) -> &alloc::vec::Vec<i64> {
        &self.scalar_indices
    }

    fn get_block_start_indices(&self) -> &alloc::vec::Vec<i64> {
        &self.block_indices
    }

    fn update_i(&mut self, i: usize, delta: nalgebra::DVector<f64>) {
        self.members[i].update(delta.as_view());
    }

    fn concrete_type_name(&self) -> &'static str {
        std::any::type_name::<Var>()
    }
}

dyn_clone::clone_trait_object!(IsVarFamily);
