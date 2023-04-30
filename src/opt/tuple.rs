use std::fmt::Debug;

trait SimpleElem: Debug {}

impl SimpleElem for usize {}

impl SimpleElem for char {}

pub trait SimpleTuple<T>: Debug {
    const N: usize;
}

impl<T: SimpleElem> SimpleTuple<T> for () {
    const N: usize = 0;
}

impl<T: SimpleElem> SimpleTuple<T> for T {
    const N: usize = 1;
}

impl<T: SimpleElem, Tail> SimpleTuple<T> for (T, Tail)
where
    Tail: SimpleTuple<T>,
{
    const N: usize = 1 + Tail::N;
}
