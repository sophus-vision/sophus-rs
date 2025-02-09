use core::fmt::Debug;
use std::any::type_name;

use as_any::Downcast;
use snafu::Snafu;
use sophus_autodiff::manifold::IsVariable;

use super::{
    var_builder::VarBuilder,
    var_family::IsVarFamily,
    VarKind,
};
use crate::variables::var_family::VarFamily;

extern crate alloc;

/// A pool of variable families
#[derive(Debug, Clone)]
pub struct VarFamilies {
    pub(crate) collection: alloc::collections::BTreeMap<String, alloc::boxed::Box<dyn IsVarFamily>>,
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
        let mut scalar_offset = 0;

        for (_name, family) in builder.families.iter_mut() {
            family.calc_start_indices(&mut scalar_offset);
        }

        Self {
            collection: builder.families,
        }
    }

    /// index
    pub fn index(&self, name: &String) -> Option<usize> {
        for (i, (family_name, _)) in self.collection.iter().enumerate() {
            if *family_name == *name {
                return Some(i);
            }
        }
        None
    }

    /// total number of free scalars
    pub fn num_free_scalars(&self) -> usize {
        let mut num = 0;

        for f in self.collection.iter() {
            num += f.1.num_free_scalars();
        }

        num
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
