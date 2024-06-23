/// Boolean mask - generalization of boolean comparison to SIMDs
pub trait IsBoolMask {
    /// Mask with all lanes set to true
    fn all_true() -> Self;

    /// Mask with all lanes set to false
    fn all_false() -> Self;

    /// Returns true if all lanes are true
    fn all(&self) -> bool;

    /// Returns true if any lane is true
    fn any(&self) -> bool;

    /// Returns the number of lanes that are true
    fn count(&self) -> usize;

    /// Returns the number of lanes
    fn lanes(&self) -> usize;

    /// and
    fn and(&self, other: &Self) -> Self;

    /// or
    fn or(&self, other: &Self) -> Self;

    /// not
    fn not(&self) -> Self;

    /// xor
    fn xor(&self, other: &Self) -> Self;

    /// Debug string - this way due to orphan rules
    fn dbg_str(&self) -> String;
}

impl IsBoolMask for bool {
    fn all_true() -> bool {
        true
    }
    fn all_false() -> bool {
        false
    }

    fn all(&self) -> bool {
        *self
    }

    fn any(&self) -> bool {
        *self
    }

    fn count(&self) -> usize {
        match *self {
            true => 1,
            false => 0,
        }
    }

    fn lanes(&self) -> usize {
        1
    }

    fn and(&self, other: &Self) -> Self {
        *self && *other
    }

    fn or(&self, other: &Self) -> Self {
        *self || *other
    }

    fn not(&self) -> Self {
        !*self
    }

    fn xor(&self, other: &Self) -> Self {
        *self ^ *other
    }

    fn dbg_str(&self) -> String {
        format!("{}", self)
    }
}
