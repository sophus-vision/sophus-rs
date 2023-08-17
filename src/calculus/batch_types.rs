use dfdx::{shapes::*, tensor::*};

pub type SharedTape = std::sync::Arc<std::sync::Mutex<OwnedTape<f64, Cpu>>>;

pub trait SophusTape :dfdx::tensor::Tape<f64, Cpu> + std::fmt::Debug + Clone{
}

impl SophusTape for NoneTape {}
impl SophusTape for SharedTape {}



// Batched scalar - with generic tape; rank1 tensor
pub type GenS<const BATCH: usize, MaybeTape> = Tensor<Rank1<BATCH>, f64, Cpu, MaybeTape>;
// Batched vector - with generic tape; rank2 tensor
pub type GenV<const BATCH: usize, const ROWS: usize, MaybeTape> =
    Tensor<Rank2<BATCH, ROWS>, f64, Cpu, MaybeTape>;
// Batched matrix - with generic tape; rank3 tensor
pub type GenM<const BATCH: usize, const ROWS: usize, const COLS: usize, MaybeTape> =
    Tensor<Rank3<BATCH, ROWS, COLS>, f64, Cpu, MaybeTape>;

// Batched scalar - no tape; rank1 tensor
pub type S<const BATCH: usize> = GenS<BATCH, NoneTape>;
// Batched vector - no tape; rank2 tensor
pub type V<const BATCH: usize, const ROWS: usize> = GenV<BATCH, ROWS, NoneTape>;
// Batched matrix - no tape; rank3 tensor
pub type M<const BATCH: usize, const ROWS: usize, const COLS: usize> =
    GenM<BATCH, ROWS, COLS, NoneTape>;

// Batched scalar - with tape; rank1 tensor
pub type TapedS<const BATCH: usize> = GenS<BATCH, SharedTape>;
// Batched vector - with tape; rank2 tensor
pub type TapedV<const BATCH: usize, const ROWS: usize> = GenV<BATCH, ROWS, SharedTape>;
// Batched matrix - with tape; rank3 tensor
pub type TapedM<const BATCH: usize, const ROWS: usize, const COLS: usize> =
    GenM<BATCH, ROWS, COLS, SharedTape>;

// Batched vector-valued function from vector - with generic tape; rank3 tensor
//   f_b: rank1 -> rank1;  V_b := f(V_b)   [b in 0..BATCH]
pub type GenVFromV<
    const BATCH: usize,
    const INROWS: usize,
    const OUTROWS: usize,
    MaybeTape,
> = Tensor<Rank3<BATCH, INROWS, OUTROWS>, f64, Cpu, MaybeTape>;
// Batched vector-valued function from matrix - with generic tape: rank4 tensor
//   f_b: rank1 -> rank2;  V_b := f(M_b)   [b in 0..BATCH]
pub type GenVFromM<
    const BATCH: usize,
    const INROWS: usize,
    const OUTROWS: usize,
    const OUTCOLS: usize,
    MaybeTape,
> = Tensor<dfdx::shapes::Rank4<BATCH, INROWS, OUTROWS, OUTCOLS>, f64, Cpu, MaybeTape>;
// Batched vector-valued function from matrix - with generic tape: rank4 tensor
//   f_b: rank2 -> rank1;  M_b := f(V_b)   [b in 0..BATCH]
pub type GenFromV<
    const BATCH: usize,
    const INROWS: usize,
    const INCOLS: usize,
    const OUTROWS: usize,
    MaybeTape,
> = Tensor<dfdx::shapes::Rank4<BATCH, INROWS, INCOLS, OUTROWS>, f64, Cpu, MaybeTape>;
// Batched vector-valued function from matrix - with generic tape: rank5 tensor
//   f_b: rank2 -> rank2;  M_b := f(M_b)   [b in 0..BATCH]
pub type GenMFromM<
    const BATCH: usize,
    const INROWS: usize,
    const INCOLS: usize,
    const OUTROWS: usize,
    const OUTCOLS: usize,
    MaybeTape,
> = Tensor<dfdx::shapes::Rank5<BATCH, INROWS, INCOLS, OUTROWS, OUTCOLS>, f64, Cpu, MaybeTape>;

// Batched vector-valued function from vector - with generic tape; rank3 tensor
//   f_b: rank1 -> rank1;  V_b := f(V_b)   [b in 0..BATCH]
pub type VFromV<const BATCH: usize, const INROWS: usize, const OUTROWS: usize> =
    Tensor<Rank3<BATCH, INROWS, OUTROWS>, f64, Cpu, NoneTape>;
// Batched vector-valued function from matrix - no tape: rank4 tensor
//   f_b: rank1 -> rank2;  V_b := f(M_b)   [b in 0..BATCH]
pub type VFromM<
    const BATCH: usize,
    const INROWS: usize,
    const OUTROWS: usize,
    const OUTCOLS: usize,
> = Tensor<dfdx::shapes::Rank4<BATCH, INROWS, OUTROWS, OUTCOLS>, f64, Cpu, NoneTape>;
// Batched vector-valued function from matrix - with generic tape: rank4 tensor
//   f_b: rank2 -> rank1;  M_b := f(V_b)   [b in 0..BATCH]
pub type MFromV<
    const BATCH: usize,
    const INROWS: usize,
    const INCOLS: usize,
    const OUTROWS: usize,
> = Tensor<dfdx::shapes::Rank4<BATCH, INROWS, INCOLS, OUTROWS>, f64, Cpu, NoneTape>;
// Batched vector-valued function from matrix - with generic tape: rank4 tensor
//   f_b: rank2 -> rank2;  M_b := f(M_b)   [b in 0..BATCH]
pub type MFromM<
    const BATCH: usize,
    const INROWS: usize,
    const INCOLS: usize,
    const OUTROWS: usize,
    const OUTCOLS: usize,
> = Tensor<dfdx::shapes::Rank5<BATCH, INROWS, INCOLS, OUTROWS, OUTCOLS>, f64, Cpu, NoneTape>;

// Batched vector-valued function from vector - with tape; rank3 tensor
//   f_b: rank1 -> rank1;  V_b := f(V_b)   [b in 0..BATCH]
pub type TapedVFromV<
    const BATCH: usize,
    const INROWS: usize,
    const OUTROWS: usize,
    MaybeTape,
> = Tensor<Rank3<BATCH, INROWS, OUTROWS>, f64, Cpu, MaybeTape>;
// Batched vector-valued function from matrix - with tape: rank4 tensor
//   f_b: rank1 -> rank2;  V_b := f(M_b)   [b in 0..BATCH]
pub type TapedVFromM<
    const BATCH: usize,
    const INROWS: usize,
    const OUTROWS: usize,
    const OUTCOLS: usize,
    MaybeTape,
> = Tensor<dfdx::shapes::Rank4<BATCH, INROWS, OUTROWS, OUTCOLS>, f64, Cpu, MaybeTape>;
// Batched vector-valued function from matrix - with tape: rank4 tensor
//   f_b: rank2 -> rank1;  M_b := f(V_b)   [b in 0..BATCH]
pub type TapedFromV<
    const BATCH: usize,
    const INROWS: usize,
    const INCOLS: usize,
    const OUTROWS: usize,
    MaybeTape,
> = Tensor<dfdx::shapes::Rank4<BATCH, INROWS, INCOLS, OUTROWS>, f64, Cpu, MaybeTape>;
// Batched vector-valued function from matrix - with tape: rank5 tensor
//   f_b: rank2 -> rank2;  M_b := f(M_b)   [b in 0..BATCH]
pub type TapedMFromM<
    const BATCH: usize,
    const INROWS: usize,
    const INCOLS: usize,
    const OUTROWS: usize,
    const OUTCOLS: usize,
    MaybeTape,
> = Tensor<dfdx::shapes::Rank5<BATCH, INROWS, INCOLS, OUTROWS, OUTCOLS>, f64, Cpu, MaybeTape>;

pub type IV<const BATCH: usize, const ROWS: usize> = Tensor<Rank2<BATCH, ROWS>, i64, Cpu>;

