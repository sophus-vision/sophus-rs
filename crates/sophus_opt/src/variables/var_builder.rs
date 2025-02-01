use core::fmt::Debug;
use sophus_autodiff::manifold::IsVariable;

use super::var_families::VarFamilies;
use super::var_family::IsVarFamily;
use super::var_family::VarFamily;

extern crate alloc;

/// Builder for variable families
#[derive(Debug, Clone)]
pub struct VarBuilder {
    pub(crate) families: alloc::collections::BTreeMap<String, alloc::boxed::Box<dyn IsVarFamily>>,
}

impl Default for VarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VarBuilder {
    /// create a new set of variable families
    pub fn new() -> Self {
        Self {
            families: alloc::collections::btree_map::BTreeMap::new(),
        }
    }

    /// add a family of variables to the pool
    pub fn add_family<Var: IsVariable + 'static>(
        mut self,
        name: &str,
        family: VarFamily<Var>,
    ) -> Self {
        self.families
            .insert(name.into(), alloc::boxed::Box::new(family));
        self
    }

    /// build the variable pool
    pub fn build(self) -> VarFamilies {
        VarFamilies::new(self)
    }
}
