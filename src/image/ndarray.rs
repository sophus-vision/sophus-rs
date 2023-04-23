use super::{dyn_arc_image::IntensityImage, layout::ImageSizeTrait, view::ImageViewTrait};

#[cfg(not(target_arch = "wasm32"))]
use numpy::{
    ndarray::{ArrayView, Dim},
    pyo3::{IntoPy, PyObject, Python},
    PyArray,
};

#[cfg(not(target_arch = "wasm32"))]
pub enum NDArrayViewEnum<'a> {
    U8(ArrayView<'a, u8, Dim<[usize; 2]>>),
    XU8(ArrayView<'a, u8, Dim<[usize; 3]>>),
    U16(ArrayView<'a, u16, Dim<[usize; 2]>>),
    XU16(ArrayView<'a, u16, Dim<[usize; 3]>>),
    F32(ArrayView<'a, f32, Dim<[usize; 2]>>),
    XF32(ArrayView<'a, f32, Dim<[usize; 3]>>),
}

#[cfg(not(target_arch = "wasm32"))]
pub trait NdarrayViewTrait {
    fn ndarray_view(&self) -> NDArrayViewEnum<'_>;
}

#[cfg(not(target_arch = "wasm32"))]
impl NdarrayViewTrait for IntensityImage {
    fn ndarray_view(&self) -> NDArrayViewEnum<'_> {
        match &self.buffer {
            super::arc_image::IntensityImageEnum::PU8(img) => NDArrayViewEnum::U8(
                ArrayView::from_shape((self.height(), self.width()), img.scalar_slice()).unwrap(),
            ),
            super::arc_image::IntensityImageEnum::PU16(img) => NDArrayViewEnum::U16(
                ArrayView::from_shape((self.height(), self.width()), img.scalar_slice()).unwrap(),
            ),
            super::arc_image::IntensityImageEnum::PF32(img) => NDArrayViewEnum::F32(
                ArrayView::from_shape((self.height(), self.width()), img.scalar_slice()).unwrap(),
            ),
            super::arc_image::IntensityImageEnum::P3U8(img) => NDArrayViewEnum::XU8(
                ArrayView::from_shape((self.height(), self.width(), 3), img.scalar_slice())
                    .unwrap(),
            ),
            super::arc_image::IntensityImageEnum::P3U16(img) => NDArrayViewEnum::XU16(
                ArrayView::from_shape((self.height(), self.width(), 3), img.scalar_slice())
                    .unwrap(),
            ),
            super::arc_image::IntensityImageEnum::P3F32(img) => NDArrayViewEnum::XF32(
                ArrayView::from_shape((self.height(), self.width(), 3), img.scalar_slice())
                    .unwrap(),
            ),
            super::arc_image::IntensityImageEnum::P4U8(img) => NDArrayViewEnum::XU8(
                ArrayView::from_shape((self.height(), self.width(), 4), img.scalar_slice())
                    .unwrap(),
            ),
            super::arc_image::IntensityImageEnum::P4U16(img) => NDArrayViewEnum::XU16(
                ArrayView::from_shape((self.height(), self.width(), 4), img.scalar_slice())
                    .unwrap(),
            ),
            super::arc_image::IntensityImageEnum::P4F32(img) => NDArrayViewEnum::XF32(
                ArrayView::from_shape((self.height(), self.width(), 4), img.scalar_slice())
                    .unwrap(),
            ),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub enum PyArrayEnum<'py> {
    U8(&'py PyArray<u8, Dim<[usize; 2]>>),
    XU8(&'py PyArray<u8, Dim<[usize; 3]>>),
    U16(&'py PyArray<u16, Dim<[usize; 2]>>),
    XU16(&'py PyArray<u16, Dim<[usize; 3]>>),
    F32(&'py PyArray<f32, Dim<[usize; 2]>>),
    XF32(&'py PyArray<f32, Dim<[usize; 3]>>),
}

#[cfg(not(target_arch = "wasm32"))]
impl<'py> IntoPy<PyObject> for PyArrayEnum<'py> {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        match self {
            PyArrayEnum::F32(py_array) => py_array.into(),
            PyArrayEnum::U8(py_array) => py_array.into(),
            PyArrayEnum::XU8(py_array) => py_array.into(),
            PyArrayEnum::U16(py_array) => py_array.into(),
            PyArrayEnum::XU16(py_array) => py_array.into(),
            PyArrayEnum::XF32(py_array) => py_array.into(),
        }
    }
}
