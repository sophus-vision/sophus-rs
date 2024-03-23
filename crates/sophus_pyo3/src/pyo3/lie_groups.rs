use numpy::PyArray1;
use numpy::PyArray2;
use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::Py;
use pyo3::Python;

use sophus_calculus::types::params::HasParams;
use sophus_lie::isometry2::Isometry2;
use sophus_lie::isometry3::Isometry3;
use sophus_lie::rotation2::Rotation2;
use sophus_lie::rotation3::Rotation3;
use sophus_lie::traits::IsTranslationProductGroup;

macro_rules! crate_py_lie_group_class {
    ($py_group: ident, $rust_group:ty, $name: literal,
      $dof:literal, $params:literal, $point:literal, $ambient:literal) => {
        /// Python wrapper for 2D isometry group
        #[pyclass(name = $name)]
        #[derive(Debug, Clone)]
        pub struct $py_group {
            /// 2D isometry group
            inner: $rust_group,
        }

        #[pymethods]
        impl $py_group {
            #[new]
            /// create 2D isometry group
            pub fn identity() -> Self {
                Self {
                    inner: <$rust_group>::identity(),
                }
            }

            #[staticmethod]
            fn exp(py: Python, tangent: &PyArray1<f64>) -> Py<Self> {
                assert_eq!(tangent.len(), $dof);
                let read_only_tangent = tangent.readonly();
                let tangent_slice = read_only_tangent.as_slice().unwrap();
                let tangent_vec = nalgebra::SVector::<f64, $dof>::from_column_slice(tangent_slice);

                let result = <$rust_group>::exp(&tangent_vec);
                Py::new(py, Self { inner: result }).unwrap()
            }

            fn log(&self, py: Python) -> Py<PyArray1<f64>> {
                let log = self.inner.log();
                PyArray1::from_slice(py, log.as_slice()).to_owned()
            }

            #[staticmethod]
            fn from_params(py: Python, params: &PyArray1<f64>) -> Py<Self> {
                assert_eq!(params.len(), $params);
                let read_only_params = params.readonly();
                let params_slice = read_only_params.as_slice().unwrap();
                let params_vec = nalgebra::SVector::<f64, $params>::from_column_slice(params_slice);

                let result = <$rust_group>::from_params(&params_vec);
                Py::new(py, Self { inner: result }).unwrap()
            }

            fn params(&self, py: Python) -> Py<PyArray1<f64>> {
                let params = self.inner.params();
                PyArray1::from_slice(py, params.as_slice()).to_owned()
            }

            fn transform(&self, py: Python, point: &PyArray1<f64>) -> Py<PyArray1<f64>> {
                assert_eq!(point.len(), $point);
                let read_only_point = point.readonly();
                let point_slice = read_only_point.as_slice().unwrap();
                let point_vec = nalgebra::SVector::<f64, $point>::from_column_slice(point_slice);

                let result = self.inner.transform(&point_vec);
                PyArray1::from_slice(py, result.fixed_rows::<$point>(0).as_slice()).to_owned()
            }

            fn inverse(&self) -> Self {
                Self {
                    inner: self.inner.inverse(),
                }
            }

            fn group_mul(&self, other: &$py_group) -> Self {
                Self {
                    inner: self.inner.group_mul(&other.inner),
                }
            }

            fn compact(&self, py: Python) -> Py<PyArray2<f64>> {
                let compact = self.inner.compact();
                PyArray1::from_slice(py, compact.as_slice())
                    .reshape([$point, $ambient])
                    .unwrap()
                    .to_owned()
            }

            fn matrix(&self, py: Python) -> Py<PyArray2<f64>> {
                let matrix = self.inner.matrix();
                PyArray1::from_slice(py, matrix.as_slice())
                    .reshape([$ambient, $ambient])
                    .unwrap()
                    .to_owned()
            }

            fn __str__(&self) -> String {
                format!("{}", self.inner.compact())
            }
        }
    };
}

crate_py_lie_group_class!(PyRotation2, Rotation2::<f64>, "Rotation2", 1, 2, 2, 2);
crate_py_lie_group_class!(PyIsometry2, Isometry2::<f64>, "Isometry2", 3, 4, 2, 3);
crate_py_lie_group_class!(PyRotation3, Rotation3::<f64>, "Rotation3", 3, 4, 3, 3);
crate_py_lie_group_class!(PyIsometry3, Isometry3::<f64>, "Isometry3", 6, 7, 3, 4);

macro_rules! augment_py_product_group_class {
    ($py_product_group: ident,  $rust_group:ty, $py_factor_group: ident, $point:literal) => {
        // second pymethods block, requires the "multiple-pymethods" feature
        #[pymethods]
        impl $py_product_group {
            #[staticmethod]
            fn from_translation_and_rotation(
                py: Python,
                translation: &PyArray1<f64>,
                rotation: $py_factor_group,
            ) -> Py<Self> {
                assert_eq!(translation.len(), $point);
                let read_only_translation = translation.readonly();
                let translation_slice = read_only_translation.as_slice().unwrap();
                let translation_vec =
                    nalgebra::SVector::<f64, $point>::from_column_slice(translation_slice);

                let result =
                    <$rust_group>::from_translation_and_rotation(&translation_vec, &rotation.inner);
                Py::new(py, Self { inner: result }).unwrap()
            }

            fn translation(&self, py: Python) -> Py<PyArray1<f64>> {
                let translation = self.inner.translation();
                PyArray1::from_slice(py, translation.as_slice()).to_owned()
            }

            fn rotation(&self) -> $py_factor_group {
                $py_factor_group {
                    inner: self.inner.rotation(),
                }
            }

            fn set_translation(&mut self, translation: &PyArray1<f64>) {
                let read_only_translation = translation.readonly();
                let translation_slice = read_only_translation.as_slice().unwrap();
                let translation_vec =
                    nalgebra::SVector::<f64, $point>::from_column_slice(translation_slice);

                self.inner.set_translation(&translation_vec);
            }

            fn set_rotation(&mut self, rotation: $py_factor_group) {
                self.inner.set_rotation(&rotation.inner);
            }
        }
    };
}

augment_py_product_group_class!(PyIsometry2, Isometry2<f64>, PyRotation2, 2);
augment_py_product_group_class!(PyIsometry3, Isometry3<f64>, PyRotation3, 3);
