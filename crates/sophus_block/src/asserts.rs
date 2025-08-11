/// assert that left is less than or equal to right
#[macro_export]
macro_rules! assert_le {
    ($left:expr, $right:expr $(,)?) => {
        if !($left <= $right) {
            panic!(
                "assertion failed: `(left <= right)`\n left: `{:?}`, right: `{:?}`",
                $left, $right
            );
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        if !($left <= $right) {
            panic!(
                "assertion failed: `(left <= right)`\n left: `{:?}`, right: `{:?}`\n{}",
                $left, $right, format_args!($($arg)+)
            );
        }
    };
}

/// assert that left is greater than or equal to right
#[macro_export]
macro_rules! assert_ge {
    ($left:expr, $right:expr $(,)?) => {
        if !($left >= $right) {
            panic!(
                "assertion failed: `(left >= right)`\n left: `{:?}`, right: `{:?}`",
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
