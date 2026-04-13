use core::fmt::Debug;
use std::any::type_name;

use as_any::Downcast;
use snafu::Snafu;
use sophus_autodiff::manifold::IsVariable;

use super::{
    VarKind,
    var_builder::VarBuilder,
    var_family::IsVarFamily,
};
use crate::variables::var_family::VarFamily;

extern crate alloc;

/// A pool of variable families
#[derive(Debug, Clone)]
pub struct VarFamilies {
    pub(crate) collection: alloc::collections::BTreeMap<String, alloc::boxed::Box<dyn IsVarFamily>>,
    /// Maps BTreeMap family index → partition index in the linear system.
    /// `usize::MAX` means the family has no partition (Conditioned or zero active vars).
    /// Free families occupy partitions 0..free_partition_count,
    /// Marginalized families occupy partitions free_partition_count..total_active_partition_count.
    pub(crate) partition_idx_by_family: alloc::vec::Vec<usize>,
    /// Maps family name → index in BTreeMap iteration order (for O(1) lookup).
    family_index_by_name: std::collections::HashMap<String, usize>,
    pub(crate) num_free_scalars: usize,
    pub(crate) num_marg_scalars: usize,
    pub(crate) free_partition_count: usize,
    pub(crate) total_active_partition_count: usize,
}

/// Errors that can occur when working with variable families
#[derive(Snafu, Debug)]
pub enum VarFamilyError {
    /// Tried to retrieve a variable family as the wrong type
    #[snafu(display(
        "Tried to retrieve `{family_name}` as type '{requested_type}',
                     but actual type is '{actual_type}"
    ))]
    TypeError {
        /// The name of the variable family
        family_name: String,
        /// The type that was requested
        requested_type: String,
        /// The actual type of the variable family
        actual_type: String,
    },
    /// Variable family not found in collection
    #[snafu(display("Variable family '{family_name}' not found in collection."))]
    UnknownFamily {
        /// The name of the variable family
        family_name: String,
    },
}

impl VarFamilies {
    pub(crate) fn new(mut builder: VarBuilder) -> Self {
        let n_families = builder.families.len();
        let mut partition_idx_by_family = alloc::vec![usize::MAX; n_families];
        let mut scalar_offset = 0usize;
        let mut next_partition = 0usize;

        // Pass 1: Free families (assign scalar indices and partition indices)
        for (i, (_name, family)) in builder.families.iter_mut().enumerate() {
            if family.get_var_kind() == VarKind::Free {
                family.calc_start_indices(&mut scalar_offset);
                partition_idx_by_family[i] = next_partition;
                next_partition += 1;
            }
        }

        // Pass 1b: Conditioned families (assign -1 scalar indices, no partition)
        for (_name, family) in builder.families.iter_mut() {
            if family.get_var_kind() == VarKind::Conditioned {
                family.calc_start_indices(&mut scalar_offset);
            }
        }

        let num_free_scalars = scalar_offset;
        let free_partition_count = next_partition;

        // Pass 2: Marginalized families (scalar indices continue after free)
        for (i, (_name, family)) in builder.families.iter_mut().enumerate() {
            if family.get_var_kind() == VarKind::Marginalized {
                family.calc_start_indices(&mut scalar_offset);
                partition_idx_by_family[i] = next_partition;
                next_partition += 1;
            }
        }

        let num_marg_scalars = scalar_offset - num_free_scalars;
        let total_active_partition_count = next_partition;

        let family_index_by_name: std::collections::HashMap<String, usize> = builder
            .families
            .keys()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        Self {
            collection: builder.families,
            partition_idx_by_family,
            family_index_by_name,
            num_free_scalars,
            num_marg_scalars,
            free_partition_count,
            total_active_partition_count,
        }
    }

    /// index
    pub fn index(&self, name: &str) -> Option<usize> {
        self.family_index_by_name.get(name).copied()
    }

    /// Returns the partition index for a given family name.
    ///
    /// This maps a family name to the partition index used in the linear system
    /// and Hessian. Returns `None` if the family doesn't exist or has no partition
    /// (e.g. all members are conditioned).
    pub fn partition_index(&self, name: &str) -> Option<usize> {
        let &i = self.family_index_by_name.get(name)?;
        let idx = self.partition_idx_by_family[i];
        if idx == usize::MAX { None } else { Some(idx) }
    }

    /// Returns the block index for a variable within its family's partition.
    ///
    /// For conditioned (fixed) variables, returns `None`.
    /// For free/marginalized variables, returns the block index within the partition
    /// (skipping conditioned members).
    pub fn block_index(&self, name: &str, var_idx: usize) -> Option<usize> {
        let family = self.collection.get(name)?;
        let block_indices = family.get_block_start_indices();
        if var_idx >= block_indices.len() {
            return None;
        }
        let idx = block_indices[var_idx];
        if idx < 0 { None } else { Some(idx as usize) }
    }

    /// total number of free scalars
    pub fn num_free_scalars(&self) -> usize {
        self.num_free_scalars
    }

    /// total number of marginalized scalars
    pub fn num_marg_scalars(&self) -> usize {
        self.num_marg_scalars
    }

    /// total number of active scalars (free + marginalized)
    pub(crate) fn num_active_scalars(&self) -> usize {
        self.num_free_scalars + self.num_marg_scalars
    }

    /// total number of active partitions (free + marginalized)
    pub(crate) fn total_active_partition_count(&self) -> usize {
        self.total_active_partition_count
    }

    /// family names
    pub fn names(&self) -> alloc::vec::Vec<String> {
        self.collection.keys().cloned().collect()
    }

    /// dimension for each free variable family
    pub fn dims(&self) -> Vec<usize> {
        let mut dims = Vec::new();
        for f in self.collection.iter() {
            dims.push(f.1.free_or_marg_dof());
        }
        dims
    }

    /// total number of free variables per family
    pub fn free_vars(&self) -> Vec<usize> {
        let mut v = Vec::new();

        for f in self.collection.iter() {
            v.push(f.1.num_free_vars());
        }

        v
    }

    /// total number of marginalized scalars
    pub fn num_of_kind(&self, var_kind: VarKind) -> usize {
        let mut num = 0;

        for f in self.collection.iter() {
            num += if f.1.get_var_kind() == var_kind { 1 } else { 0 }
        }

        num
    }

    /// update the variables in-place (called during optimization)
    pub(crate) fn update(&self, delta: &nalgebra::DVector<f64>) -> VarFamilies {
        let mut updated = self.clone();
        for family in updated.collection.iter_mut() {
            family.1.update(delta.as_view());
        }
        updated
    }

    /// retrieve a variable family by name
    pub fn get<T: IsVarFamily>(&self, name: &str) -> Result<&T, VarFamilyError> {
        // 1) Look up the family in the map:
        let family = self
            .collection
            .get(name)
            .ok_or_else(|| VarFamilyError::UnknownFamily {
                family_name: name.to_string(),
            })?;

        // 2) Attempt downcast:
        family
            .as_ref()
            .downcast_ref::<T>()
            .ok_or_else(|| VarFamilyError::TypeError {
                family_name: name.to_string(),
                requested_type: type_name::<T>().to_owned(),
                actual_type: family.concrete_type_name().to_owned(),
            })
    }

    /// retrieve family members by family name
    ///
    /// Panics if the family does not exist, or the specified type is not correct
    pub fn get_members<T: IsVariable + 'static>(&self, name: &str) -> alloc::vec::Vec<T> {
        let family = self
            .collection
            .get(name)
            .ok_or_else(|| VarFamilyError::UnknownFamily {
                family_name: name.to_string(),
            })
            .unwrap();
        family
            .as_ref()
            .downcast_ref::<VarFamily<T>>()
            .ok_or_else(|| VarFamilyError::TypeError {
                family_name: name.to_string(),
                requested_type: type_name::<T>().to_owned(),
                actual_type: family.concrete_type_name().to_owned(),
            })
            .unwrap()
            .members
            .clone()
    }
}
