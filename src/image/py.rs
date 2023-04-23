use super::dyn_arc_image::IntensityImage;
use super::ndarray::NDArrayViewEnum;
use super::ndarray::PyArrayEnum;
use super::pixel::PixelTag;
use crate::image::dyn_mut_image::MutIntensityImage;
use crate::image::layout::ImageSize;
use crate::image::ndarray::NdarrayViewTrait;
use numpy::ToPyArray;
use pyo3::pymethods;
use pyo3::Python;

#[pymethods]
impl ImageSize {
    #[new]
    pub fn pyo3_new(width: usize, height: usize) -> Self {
        ImageSize { width, height }
    }
}

#[pymethods]
impl MutIntensityImage {
    #[new]
    fn pyo3_new(size: ImageSize, tag: PixelTag) -> Self {
        MutIntensityImage::with_size_and_tag(size, tag)
    }
}

#[pymethods]
impl IntensityImage {
    #[new]
    fn pyo3_new(img: MutIntensityImage) -> Self {
        IntensityImage::from_dyn_mut(img)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn pyarray<'py>(&self, py: Python<'py>) -> PyArrayEnum<'py> {
        match self.ndarray_view() {
            NDArrayViewEnum::U8(a) => PyArrayEnum::U8(a.to_pyarray(py)),
            NDArrayViewEnum::XU8(a) => PyArrayEnum::XU8(a.to_pyarray(py)),
            NDArrayViewEnum::U16(a) => PyArrayEnum::U16(a.to_pyarray(py)),
            NDArrayViewEnum::XU16(a) => PyArrayEnum::XU16(a.to_pyarray(py)),
            NDArrayViewEnum::F32(a) => PyArrayEnum::F32(a.to_pyarray(py)),
            NDArrayViewEnum::XF32(a) => PyArrayEnum::XF32(a.to_pyarray(py)),
        }
    }
}
