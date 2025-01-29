use sophus_autodiff::manifold::IsVariable;

use super::var_families::VarFamilies;
use super::var_family::VarFamily;
use super::VarKind;

extern crate alloc;

/// Tuple of variables (one for each argument of the cost function)
pub trait IsVarTuple<const NUM_ARGS: usize>: Send + Sync + 'static {
    /// number of degrees of freedom for each variable
    const DOF_T: [usize; NUM_ARGS];

    /// Tuple variable families
    type VarFamilyTupleRef<'a>: Send + Sync;

    /// reference to the variable family tuple
    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; NUM_ARGS],
    ) -> Self::VarFamilyTupleRef<'_>;

    /// extract the variables from the family tuple
    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; NUM_ARGS]) -> Self;

    /// return the variable kind for each variable (one for each argument of the cost function)
    fn var_kind_array(families: &VarFamilies, names: [String; NUM_ARGS]) -> [VarKind; NUM_ARGS];
}

impl<M0: IsVariable + 'static + Send + Sync> IsVarTuple<1> for M0 {
    const DOF_T: [usize; 1] = [M0::DOF];
    type VarFamilyTupleRef<'a> = &'a VarFamily<M0>;

    fn var_kind_array(families: &VarFamilies, names: [String; 1]) -> [VarKind; 1] {
        [families.collection.get(&names[0]).unwrap().get_var_kind()]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 1],
    ) -> Self::VarFamilyTupleRef<'_> {
        families.get::<VarFamily<M0>>(names[0].clone())
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 1]) -> Self {
        family_tuple.members[ids[0]].clone()
    }
}

impl<M0: IsVariable + 'static + Send + Sync, M1: IsVariable + 'static + Send + Sync> IsVarTuple<2>
    for (M0, M1)
{
    const DOF_T: [usize; 2] = [M0::DOF, M1::DOF];
    type VarFamilyTupleRef<'a> = (&'a VarFamily<M0>, &'a VarFamily<M1>);

    fn var_kind_array(families: &VarFamilies, names: [String; 2]) -> [VarKind; 2] {
        [
            families.collection.get(&names[0]).unwrap().get_var_kind(),
            families.collection.get(&names[1]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 2],
    ) -> Self::VarFamilyTupleRef<'_> {
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

impl<
        M0: IsVariable + 'static + Send + Sync,
        M1: IsVariable + 'static + Send + Sync,
        M2: IsVariable + 'static + Send + Sync,
    > IsVarTuple<3> for (M0, M1, M2)
{
    const DOF_T: [usize; 3] = [M0::DOF, M1::DOF, M2::DOF];
    type VarFamilyTupleRef<'a> = (&'a VarFamily<M0>, &'a VarFamily<M1>, &'a VarFamily<M2>);

    fn var_kind_array(families: &VarFamilies, names: [String; 3]) -> [VarKind; 3] {
        [
            families.collection.get(&names[0]).unwrap().get_var_kind(),
            families.collection.get(&names[1]).unwrap().get_var_kind(),
            families.collection.get(&names[2]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 3],
    ) -> Self::VarFamilyTupleRef<'_> {
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

impl<
        M0: IsVariable + 'static + Send + Sync,
        M1: IsVariable + 'static + Send + Sync,
        M2: IsVariable + 'static + Send + Sync,
        M3: IsVariable + 'static + Send + Sync,
    > IsVarTuple<4> for (M0, M1, M2, M3)
{
    const DOF_T: [usize; 4] = [M0::DOF, M1::DOF, M2::DOF, M3::DOF];
    type VarFamilyTupleRef<'a> = (
        &'a VarFamily<M0>,
        &'a VarFamily<M1>,
        &'a VarFamily<M2>,
        &'a VarFamily<M3>,
    );

    fn var_kind_array(families: &VarFamilies, names: [String; 4]) -> [VarKind; 4] {
        [
            families.collection.get(&names[0]).unwrap().get_var_kind(),
            families.collection.get(&names[1]).unwrap().get_var_kind(),
            families.collection.get(&names[2]).unwrap().get_var_kind(),
            families.collection.get(&names[3]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 4],
    ) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
            families.get::<VarFamily<M2>>(names[2].clone()),
            families.get::<VarFamily<M3>>(names[3].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 4]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
            family_tuple.2.members[ids[2]].clone(),
            family_tuple.3.members[ids[3]].clone(),
        )
    }
}

impl<
        M0: IsVariable + 'static + Send + Sync,
        M1: IsVariable + 'static + Send + Sync,
        M2: IsVariable + 'static + Send + Sync,
        M3: IsVariable + 'static + Send + Sync,
        M4: IsVariable + 'static + Send + Sync,
    > IsVarTuple<5> for (M0, M1, M2, M3, M4)
{
    const DOF_T: [usize; 5] = [M0::DOF, M1::DOF, M2::DOF, M3::DOF, M4::DOF];
    type VarFamilyTupleRef<'a> = (
        &'a VarFamily<M0>,
        &'a VarFamily<M1>,
        &'a VarFamily<M2>,
        &'a VarFamily<M3>,
        &'a VarFamily<M4>,
    );

    fn var_kind_array(families: &VarFamilies, names: [String; 5]) -> [VarKind; 5] {
        [
            families.collection.get(&names[0]).unwrap().get_var_kind(),
            families.collection.get(&names[1]).unwrap().get_var_kind(),
            families.collection.get(&names[2]).unwrap().get_var_kind(),
            families.collection.get(&names[3]).unwrap().get_var_kind(),
            families.collection.get(&names[4]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 5],
    ) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
            families.get::<VarFamily<M2>>(names[2].clone()),
            families.get::<VarFamily<M3>>(names[3].clone()),
            families.get::<VarFamily<M4>>(names[4].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 5]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
            family_tuple.2.members[ids[2]].clone(),
            family_tuple.3.members[ids[3]].clone(),
            family_tuple.4.members[ids[4]].clone(),
        )
    }
}

impl<
        M0: IsVariable + 'static + Send + Sync,
        M1: IsVariable + 'static + Send + Sync,
        M2: IsVariable + 'static + Send + Sync,
        M3: IsVariable + 'static + Send + Sync,
        M4: IsVariable + 'static + Send + Sync,
        M5: IsVariable + 'static + Send + Sync,
    > IsVarTuple<6> for (M0, M1, M2, M3, M4, M5)
{
    const DOF_T: [usize; 6] = [M0::DOF, M1::DOF, M2::DOF, M3::DOF, M4::DOF, M5::DOF];
    type VarFamilyTupleRef<'a> = (
        &'a VarFamily<M0>,
        &'a VarFamily<M1>,
        &'a VarFamily<M2>,
        &'a VarFamily<M3>,
        &'a VarFamily<M4>,
        &'a VarFamily<M5>,
    );

    fn var_kind_array(families: &VarFamilies, names: [String; 6]) -> [VarKind; 6] {
        [
            families.collection.get(&names[0]).unwrap().get_var_kind(),
            families.collection.get(&names[1]).unwrap().get_var_kind(),
            families.collection.get(&names[2]).unwrap().get_var_kind(),
            families.collection.get(&names[3]).unwrap().get_var_kind(),
            families.collection.get(&names[4]).unwrap().get_var_kind(),
            families.collection.get(&names[5]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 6],
    ) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
            families.get::<VarFamily<M2>>(names[2].clone()),
            families.get::<VarFamily<M3>>(names[3].clone()),
            families.get::<VarFamily<M4>>(names[4].clone()),
            families.get::<VarFamily<M5>>(names[5].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 6]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
            family_tuple.2.members[ids[2]].clone(),
            family_tuple.3.members[ids[3]].clone(),
            family_tuple.4.members[ids[4]].clone(),
            family_tuple.5.members[ids[5]].clone(),
        )
    }
}

impl<
        M0: IsVariable + 'static + Send + Sync,
        M1: IsVariable + 'static + Send + Sync,
        M2: IsVariable + 'static + Send + Sync,
        M3: IsVariable + 'static + Send + Sync,
        M4: IsVariable + 'static + Send + Sync,
        M5: IsVariable + 'static + Send + Sync,
        M6: IsVariable + 'static + Send + Sync,
    > IsVarTuple<7> for (M0, M1, M2, M3, M4, M5, M6)
{
    const DOF_T: [usize; 7] = [
        M0::DOF,
        M1::DOF,
        M2::DOF,
        M3::DOF,
        M4::DOF,
        M5::DOF,
        M6::DOF,
    ];
    type VarFamilyTupleRef<'a> = (
        &'a VarFamily<M0>,
        &'a VarFamily<M1>,
        &'a VarFamily<M2>,
        &'a VarFamily<M3>,
        &'a VarFamily<M4>,
        &'a VarFamily<M5>,
        &'a VarFamily<M6>,
    );

    fn var_kind_array(families: &VarFamilies, names: [String; 7]) -> [VarKind; 7] {
        [
            families.collection.get(&names[0]).unwrap().get_var_kind(),
            families.collection.get(&names[1]).unwrap().get_var_kind(),
            families.collection.get(&names[2]).unwrap().get_var_kind(),
            families.collection.get(&names[3]).unwrap().get_var_kind(),
            families.collection.get(&names[4]).unwrap().get_var_kind(),
            families.collection.get(&names[5]).unwrap().get_var_kind(),
            families.collection.get(&names[6]).unwrap().get_var_kind(),
        ]
    }

    fn ref_var_family_tuple(
        families: &VarFamilies,
        names: [String; 7],
    ) -> Self::VarFamilyTupleRef<'_> {
        (
            families.get::<VarFamily<M0>>(names[0].clone()),
            families.get::<VarFamily<M1>>(names[1].clone()),
            families.get::<VarFamily<M2>>(names[2].clone()),
            families.get::<VarFamily<M3>>(names[3].clone()),
            families.get::<VarFamily<M4>>(names[4].clone()),
            families.get::<VarFamily<M5>>(names[5].clone()),
            families.get::<VarFamily<M6>>(names[6].clone()),
        )
    }

    fn extract(family_tuple: &Self::VarFamilyTupleRef<'_>, ids: [usize; 7]) -> Self {
        (
            family_tuple.0.members[ids[0]].clone(),
            family_tuple.1.members[ids[1]].clone(),
            family_tuple.2.members[ids[2]].clone(),
            family_tuple.3.members[ids[3]].clone(),
            family_tuple.4.members[ids[4]].clone(),
            family_tuple.5.members[ids[5]].clone(),
            family_tuple.6.members[ids[6]].clone(),
        )
    }
}
