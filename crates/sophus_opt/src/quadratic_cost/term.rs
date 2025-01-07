use core::ops::Range;

extern crate alloc;

/// (Unevaluated) term of the cost function
pub trait IsTerm<const N: usize>: Send + Sync + 'static {
    /// associated constants such as measurements, etc.
    type Constants;

    /// one DOF for each argument
    const DOF_TUPLE: [i64; N];

    /// reference to the constants
    fn c_ref(&self) -> &Self::Constants;

    /// one index (into the variable family) for each argument
    fn idx_ref(&self) -> &[usize; N];
}

/// (Unevaluated) cost
#[derive(Debug, Clone)]
pub struct Terms<const NUM_ARGS: usize, Constants, Term: IsTerm<NUM_ARGS, Constants = Constants>> {
    /// one variable family name for each argument
    pub family_names: [alloc::string::String; NUM_ARGS],
    /// collection of unevaluated terms
    pub collection: alloc::vec::Vec<Term>,
    pub(crate) reduction_ranges: Option<alloc::vec::Vec<Range<usize>>>,
}

impl<const NUM_ARGS: usize, Constants, Term: IsTerm<NUM_ARGS, Constants = Constants>>
    Terms<NUM_ARGS, Constants, Term>
{
    /// Create a new set of terms
    pub fn new(
        family_names: [alloc::string::String; NUM_ARGS],
        terms: alloc::vec::Vec<Term>,
    ) -> Self {
        Terms {
            family_names,
            collection: terms,
            reduction_ranges: None,
        }
    }
}
