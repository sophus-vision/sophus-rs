/// assert that left is less than right
#[macro_export]
macro_rules! assert_lt {
    ($left:expr, $right:expr $(,)?) => {
        if !($left < $right) {
            panic!(
                "assertion failed: `(left < right)`\n left: `{:?}`, right: `{:?}`",
                $left, $right
            );
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        if !($left < $right) {
            panic!(
                "assertion failed: `(left < right)`\n left: `{:?}`, right: `{:?}`\n{}",
                $left, $right, format_args!($($arg)+)
            );
        }
    };
}

/// assert that left is greater than right
#[macro_export]
macro_rules! assert_gt {
    ($left:expr, $right:expr $(,)?) => {
        if !($left > $right) {
            panic!(
                "assertion failed: `(left > right)`\n left: `{:?}`, right: `{:?}`",
                $left, $right
            );
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        if !($left >= $right) {
            panic!(
                "assertion failed: `(left >= right)`\n left: `{:?}`, right: `{:?}`\n{}",
                $left, $right, format_args!($($arg)+)
            );
        }
    };
}

/// assert that left is less than or equal to right
#[macro_export]
macro_rules! assert_le {
    ($left:expr, $right:expr $(,)?) => {{
        let (left, right) = (&$left, &$right);
        match $crate::asserts::_assert_le_partial_cmp(left, right) {
            Ok(()) => {}
            Err(reason) => panic!(
                "assertion failed: `(left <= right)`\n left: `{:?}`, right: `{:?}`\n{}",
                left, right, reason
            ),
        }
    }};
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let (left, right) = (&$left, &$right);
        match $crate::asserts::_assert_le_partial_cmp(left, right) {
            Ok(()) => {}
            Err(reason) => panic!(
                "assertion failed: `(left <= right)`\n left: `{:?}`, right: `{:?}`\n{}\n{}",
                left, right, reason, format_args!($($arg)+)
            ),
        }
    }};
}

/// assert that left is greater than or equal to right
#[macro_export]
macro_rules! assert_ge {
    ($left:expr, $right:expr $(,)?) => {{
        let (left, right) = (&$left, &$right);
        match $crate::asserts::_assert_ge_partial_cmp(left, right) {
            Ok(()) => {}
            Err(reason) => panic!(
                "assertion failed: `(left >= right)`\n left: `{:?}`, right: `{:?}`\n{}",
                left, right, reason
            ),
        }
    }};
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let (left, right) = (&$left, &$right);
        match $crate::asserts::_assert_ge_partial_cmp(left, right) {
            Ok(()) => {}
            Err(reason) => panic!(
                "assertion failed: `(left >= right)`\n left: `{:?}`, right: `{:?}`\n{}\n{}",
                left, right, reason, format_args!($($arg)+)
            ),
        }
    }};
}

#[doc(hidden)]
pub fn _assert_le_partial_cmp<T: PartialOrd>(l: &T, r: &T) -> Result<(), &'static str> {
    use core::cmp::Ordering::*;
    match l.partial_cmp(r) {
        Some(Less) | Some(Equal) => Ok(()),
        Some(Greater) => Err("left is greater than right"),
        None => Err("values are incomparable (PartialOrd::partial_cmp returned None)"),
    }
}

#[doc(hidden)]
pub fn _assert_ge_partial_cmp<T: PartialOrd>(l: &T, r: &T) -> Result<(), &'static str> {
    use core::cmp::Ordering::*;
    match l.partial_cmp(r) {
        Some(Greater) | Some(Equal) => Ok(()),
        Some(Less) => Err("left is less than right"),
        None => Err("values are incomparable (PartialOrd::partial_cmp returned None)"),
    }
}

/// assert that left is less than or equal to right
#[macro_export]
macro_rules! debug_assert_le {
    ($($args:tt)*) => {
        if cfg!(debug_assertions) {
            use $crate::assert_le;
            assert_le!($($args)*);
        }
    };
}

/// assert that left is greater than or equal to right
#[macro_export]
macro_rules! debug_assert_ge {
    ($($args:tt)*) => {
        if cfg!(debug_assertions) {
            use $crate::assert_ge;
            assert_ge!($($args)*);
        }
    };
}
