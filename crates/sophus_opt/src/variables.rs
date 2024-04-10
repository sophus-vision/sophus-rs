use crate::prelude::*;
use dyn_clone::DynClone;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry2;
use sophus_lie::Isometry3;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Debug;

/// Variable kind
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum VarKind {
    /// free variable (will be updated during optimization)
    Free,
    /// conditioned variable (will not be fixed during optimization)
    Conditioned,
    /// marginalized variable (will be updated during using the Schur complement trick)
    Marginalized,
}

///A decision variables
pub trait IsVariable: Clone + Debug {
    /// number of degrees of freedom
    const DOF: usize;

    /// update the variable in-place (called during optimization)
    fn update(&mut self, delta: nalgebra::DVectorView<f64>);
}

/// Tuple of variables (one for each argument of the cost function)
pub trait IsVarTuple<const NUM_ARGS: usize> {
    /// number of degrees of freedom for each variable
    const DOF_T: [usize; NUM_ARGS];

    /// Tuple variable families
    type VarFamilyTupleRef<'a>;

    /// reference to the variable family tuple
    fn ref_var_family_tuple(
        families: &VarPool,
        names: [String; NUM_ARGS],
    ) -> Self::VarFamilyTupleRef<'_>;

    /// extract the variables from the family tuple
    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; NUM_ARGS]) -> Self;

    /// return the variable kind for each variable (one for each argument of the cost function)
    fn var_kind_array(families: &VarPool, names: [String; NUM_ARGS]) -> [VarKind; NUM_ARGS];
}

impl<M0: IsVariable + 'static> IsVarTuple<1> for M0 {
    const DOF_T: [usize; 1] = [M0::DOF];
    type VarFamilyTupleRef<'a> = &'a VarFamily<M0>;

    fn var_kind_array(families: &VarPool, names: [String; 1]) -> [VarKind; 1] {
        [families.families.get(&names[0]).unwrap().get_var_kind()]
    }

    fn ref_var_family_tuple(families: &VarPool, names: [String; 1]) -> Self::VarFamilyTupleRef<'_> {
        families.get::<VarFamily<M0>>(names[0].clone())
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 1]) -> Self {
        family_tuple.members[ids[0]].clone()
    }
}

impl<M0: IsVariable + 'static, M1: IsVariable + 'static> IsVarTuple<2> for (M0, M1) {
    const DOF_T: [usize; 2] = [M0::DOF, M1::DOF];
    type VarFamilyTupleRef<'a> = (&'a VarFamily<M0>, &'a VarFamily<M1>);

    fn var_kind_array(families: &VarPool, names: [String; 2]) -> [VarKind; 2] {
        [
            families.families.get(&names[0]).unwrap().get_var_kind(),
            families.families.get(&names[1]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(families: &VarPool, names: [String; 2]) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 2]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
        )
    }
}

impl<M0: IsVariable + 'static, M1: IsVariable + 'static, M2: IsVariable + 'static> IsVarTuple<3>
    for (M0, M1, M2)
{
    const DOF_T: [usize; 3] = [M0::DOF, M1::DOF, M2::DOF];
    type VarFamilyTupleRef<'a> = (&'a VarFamily<M0>, &'a VarFamily<M1>, &'a VarFamily<M2>);

    fn var_kind_array(families: &VarPool, names: [String; 3]) -> [VarKind; 3] {
        [
            families.families.get(&names[0]).unwrap().get_var_kind(),
            families.families.get(&names[1]).unwrap().get_var_kind(),
            families.families.get(&names[2]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(families: &VarPool, names: [String; 3]) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
            families.get::<VarFamily<M2>>(names[2].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 3]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
            family_tuple.2.members[ids[2]].clone(),
        )
    }
}

impl<const N: usize> IsVariable for VecF64<N> {
    const DOF: usize = N;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        for d in 0..Self::DOF {
            self[d] += delta[d];
        }
    }
}

impl IsVariable for Isometry2<f64, 1> {
    const DOF: usize = 3;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let mut delta_vec = VecF64::<3>::zeros();
        for d in 0..<Self as IsVariable>::DOF {
            delta_vec[d] = delta[d];
        }
        self.set_params(
            (Isometry2::<f64, 1>::group_mul(&Isometry2::<f64, 1>::exp(&delta_vec), &self.clone()))
                .params(),
        );
    }
}

impl IsVariable for Isometry3<f64, 1> {
    const DOF: usize = 6;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let mut delta_vec = VecF64::<6>::zeros();
        for d in 0..<Self as IsVariable>::DOF {
            delta_vec[d] = delta[d];
        }
        self.set_params(
            (Isometry3::<f64, 1>::group_mul(&Isometry3::<f64, 1>::exp(&delta_vec), &self.clone()))
                .params(),
        );
    }
}

/// A generic family of variables
///
/// A list of variables of the same nature (e.g., a list of 3D points, a list of 2D isometries, etc.)
#[derive(Debug, Clone)]
pub struct VarFamily<Var: IsVariable> {
    kind: VarKind,
    /// members of the family
    pub members: Vec<Var>,
    constant_members: HashMap<usize, ()>,
    start_indices: Vec<i64>,
}

impl<Var: IsVariable> VarFamily<Var> {
    /// create a new variable family a variable kind (free, conditioned, ...) and a list of members
    pub fn new(kind: VarKind, members: Vec<Var>) -> Self {
        Self::new_with_const_ids(kind, members, HashMap::new())
    }

    /// Create a new variable family a variable kind (free, conditioned, ...), a list of members and a list of constant members
    ///
    /// The constant members are not updated during optimization (no matter the variable kind)
    pub fn new_with_const_ids(
        kind: VarKind,
        members: Vec<Var>,
        constant_members: HashMap<usize, ()>,
    ) -> Self {
        VarFamily {
            kind,
            members,
            constant_members,
            start_indices: vec![],
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

    /// total number of free scalars (#free variables * #DOF)
    fn num_free_scalars(&self) -> usize;

    /// start indices in linear system of equations
    fn calc_start_indices(&mut self, offset: &mut usize);

    /// total number of marginalized scalars (#marginalized variables * #DOF)
    fn num_marg_scalars(&self) -> usize;

    /// start indices in linear system of equations
    fn get_start_indices(&self) -> &Vec<i64>;

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
}

impl<Var: IsVariable + 'static> IsVarFamily for VarFamily<Var> {
    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let dof = self.free_or_marg_dof();

        assert_eq!(self.start_indices.len(), self.members.len());

        for i in 0..self.start_indices.len() {
            let start_idx = self.start_indices[i];
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
    fn calc_start_indices(&mut self, inout_offset: &mut usize) {
        assert_eq!(
            self.start_indices.len(),
            0,
            "Ths function must ony called once"
        );

        match self.get_var_kind() {
            VarKind::Free => {
                let mut indices = vec![];
                let mut idx: usize = *inout_offset;
                for i in 0..self.members.len() {
                    if self.constant_members.contains_key(&i) {
                        indices.push(-1);
                    } else {
                        indices.push(idx as i64);
                        idx += Var::DOF;
                    }
                }
                *inout_offset = idx;

                assert_eq!(indices.len(), self.members.len());
                self.start_indices = indices;
            }
            VarKind::Conditioned => {
                let mut indices = vec![];

                for _i in 0..self.members.len() {
                    indices.push(-1);
                }
                assert_eq!(indices.len(), self.members.len());
                self.start_indices = indices;
            }
            VarKind::Marginalized => {
                let mut indices = vec![];

                for _i in 0..self.members.len() {
                    indices.push(-2);
                }
                assert_eq!(indices.len(), self.members.len());
                self.start_indices = indices;
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

    fn get_start_indices(&self) -> &Vec<i64> {
        &self.start_indices
    }

    fn update_i(&mut self, i: usize, delta: nalgebra::DVector<f64>) {
        self.members[i].update(delta.as_view());
    }
}

dyn_clone::clone_trait_object!(IsVarFamily);

/// Builder for the variable pool
#[derive(Debug, Clone)]
pub struct VarPoolBuilder {
    families: std::collections::BTreeMap<String, Box<dyn IsVarFamily>>,
}

impl Default for VarPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VarPoolBuilder {
    /// create a new variable pool builder
    pub fn new() -> Self {
        Self {
            families: BTreeMap::new(),
        }
    }

    /// add a family of variables to the pool
    pub fn add_family<S: Into<String>, Var: IsVariable + 'static>(
        mut self,
        name: S,
        family: VarFamily<Var>,
    ) -> Self {
        self.families.insert(name.into(), Box::new(family));
        self
    }

    /// build the variable pool
    pub fn build(self) -> VarPool {
        VarPool::new(self)
    }
}

/// A pool of variable families
#[derive(Debug, Clone)]
pub struct VarPool {
    pub(crate) families: std::collections::BTreeMap<String, Box<dyn IsVarFamily>>,
}

impl VarPool {
    fn new(mut builder: VarPoolBuilder) -> Self {
        let mut offset = 0;
        for (_name, family) in builder.families.iter_mut() {
            family.calc_start_indices(&mut offset);
        }

        Self {
            families: builder.families,
        }
    }
}

impl VarPool {
    /// total number of free scalars
    pub fn num_free_params(&self) -> usize {
        let mut num = 0;

        for f in self.families.iter() {
            num += f.1.num_free_scalars();
        }

        num
    }

    /// total number of marginalized scalars
    pub fn num_of_kind(&self, var_kind: VarKind) -> usize {
        let mut num = 0;

        for f in self.families.iter() {
            num += if f.1.get_var_kind() == var_kind { 1 } else { 0 }
        }

        num
    }

    /// update the variables in-place (called during optimization)
    pub(crate) fn update(&self, delta: nalgebra::DVector<f64>) -> VarPool {
        let mut updated = self.clone();
        for family in updated.families.iter_mut() {
            family.1.update(delta.as_view());
        }
        updated
    }

    /// retrieve a variable family by name
    ///
    /// Panics if the family does not exist, or the specified type is not correct
    pub fn get<T: IsVarFamily>(&self, name: String) -> &T {
        as_any::Downcast::downcast_ref::<T>(self.families.get(&name).unwrap().as_ref()).unwrap()
    }

    /// retrieve family members by family name
    ///
    /// Panics if the family does not exist, or the specified type is not correct
    pub fn get_members<T: IsVariable + 'static>(&self, name: String) -> Vec<T> {
        as_any::Downcast::downcast_ref::<VarFamily<T>>(self.families.get(&name).unwrap().as_ref())
            .unwrap()
            .members
            .clone()
    }
}
