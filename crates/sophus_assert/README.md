## Assert macros

Comparison assertion macros for `<`, `>`, `<=`, `>=`:

- `assert_lt!`, `assert_gt!`, `assert_le!`, `assert_ge!`
- `debug_assert_lt!`, `debug_assert_gt!`, `debug_assert_le!`, `debug_assert_ge!`

The `_le` and `_ge` variants use `PartialOrd` and produce clear error messages
when values are incomparable (e.g. `NaN`).

## Integration with sophus-rs

This crate is part of the [sophus umbrella crate](https://crates.io/crates/sophus).
