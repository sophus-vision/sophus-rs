use crate::pyo3::errors::check_array1_dim_impl;
use crate::pyo3::errors::PyArray1DimMismatch;

use sophus_calculus::types::params::HasParams;
use sophus_lie::isometry2::Isometry2;
use sophus_lie::isometry3::Isometry3;
use sophus_lie::rotation2::Rotation2;
use sophus_lie::rotation3::Rotation3;
use sophus_lie::traits::IsTranslationProductGroup;

use numpy::PyArray1;
use numpy::PyArray2;
use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::Py;
use pyo3::Python;

macro_rules! check_array1_dim {
    ($array:expr, $expected:expr) => {
        check_array1_dim_impl($array, $expected, file!(), line!())
    };
}

macro_rules! crate_py_lie_group_class {
    ($py_group: ident, $rust_group:ty, $name: literal,
      $dof:literal, $params:literal, $point:literal, $ambient:literal) => {
        /// Python wrapper for python group
        #[pyclass(name = $name)]
        #[derive(Debug, Clone)]
        pub struct $py_group {
            /// wrapped rust group
            inner: $rust_group,
        }

        #[pymethods]
        impl $py_group {
            #[staticmethod]
            fn ad(
                tangent: &PyArray1<f64>,
                py: Python,
            ) -> Result<Py<PyArray2<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(tangent, $dof)?;
                let read_only_tangent = tangent.readonly();
                let tangent_slice = read_only_tangent.as_slice().unwrap();
                let tangent_vec = nalgebra::SVector::<f64, $dof>::from_column_slice(tangent_slice);

                Ok(
                    PyArray1::from_slice(py, <$rust_group>::ad(&tangent_vec).as_slice())
                        .reshape([$ambient, $ambient])
                        .unwrap()
                        .to_owned(),
                )
            }

            fn adj(&self, py: Python) -> Py<PyArray2<f64>> {
                let adj = self.inner.adj();
                PyArray1::from_slice(py, adj.as_slice())
                    .reshape([$dof, $dof])
                    .unwrap()
                    .to_owned()
            }

            fn compact(&self, py: Python) -> Py<PyArray2<f64>> {
                let compact = self.inner.compact();
                PyArray1::from_slice(py, compact.as_slice())
                    .reshape([$point, $ambient])
                    .unwrap()
                    .to_owned()
            }

            #[staticmethod]
            fn da_a_mul_b(a: &Self, b: &Self, py: Python) -> Py<PyArray2<f64>> {
                let result = <$rust_group>::da_a_mul_b(&a.inner, &b.inner);
                PyArray1::from_slice(py, result.as_slice())
                    .reshape([$params, $params])
                    .unwrap()
                    .to_owned()
            }

            #[staticmethod]
            fn db_a_mul_b(a: &Self, b: &Self, py: Python) -> Py<PyArray2<f64>> {
                let result = <$rust_group>::db_a_mul_b(&a.inner, &b.inner);
                PyArray1::from_slice(py, result.as_slice())
                    .reshape([$params, $params])
                    .unwrap()
                    .to_owned()
            }

            #[staticmethod]
            fn dx_exp(
                tangent: &PyArray1<f64>,
                py: Python,
            ) -> Result<Py<PyArray2<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(tangent, $dof)?;
                let read_only_tangent = tangent.readonly();
                let tangent_slice = read_only_tangent.as_slice().unwrap();
                let tangent_vec = nalgebra::SVector::<f64, $dof>::from_column_slice(tangent_slice);

                let result = <$rust_group>::dx_exp(&tangent_vec);
                Ok(PyArray1::from_slice(py, result.as_slice())
                    .reshape([$params, $dof])
                    .unwrap()
                    .to_owned())
            }

            #[staticmethod]
            fn dx_exp_x_at_0(py: Python) -> Py<PyArray2<f64>> {
                let result = <$rust_group>::dx_exp_x_at_0();
                PyArray1::from_slice(py, result.as_slice())
                    .reshape([$params, $dof])
                    .unwrap()
                    .to_owned()
            }

            #[staticmethod]
            fn dx_exp_x_times_point_at_0(
                point: &PyArray1<f64>,
                py: Python,
            ) -> Result<Py<PyArray2<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(point, $point)?;
                let read_only_point = point.readonly();
                let point_slice = read_only_point.as_slice().unwrap();
                let point_vec = nalgebra::SVector::<f64, $point>::from_column_slice(point_slice);

                let result = <$rust_group>::dx_exp_x_times_point_at_0(&point_vec);
                Ok(PyArray1::from_slice(py, result.as_slice())
                    .reshape([$params, $point])
                    .unwrap()
                    .to_owned())
            }

            #[staticmethod]
            fn dx_log_a_exp_x_b_at_0(a: &Self, b: &Self, py: Python) -> Py<PyArray2<f64>> {
                let result = <$rust_group>::dx_log_a_exp_x_b_at_0(&a.inner, &b.inner);
                PyArray1::from_slice(py, result.as_slice())
                    .reshape([$dof, $dof])
                    .unwrap()
                    .to_owned()
            }

            #[staticmethod]
            fn dx_log_x(
                params: &PyArray1<f64>,
                py: Python,
            ) -> Result<Py<PyArray2<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(params, $params)?;
                let read_only_params = params.readonly();
                let params_slice = read_only_params.as_slice().unwrap();
                let params_vec = nalgebra::SVector::<f64, $params>::from_column_slice(params_slice);
                let result = <$rust_group>::dx_log_x(&params_vec);
                Ok(PyArray1::from_slice(py, result.as_slice())
                    .reshape([$dof, $params])
                    .unwrap()
                    .to_owned())
            }

            #[staticmethod]
            fn exp(tangent: &PyArray1<f64>) -> Result<Self, PyArray1DimMismatch> {
                check_array1_dim!(tangent, $dof)?;
                let read_only_tangent = tangent.readonly();
                let tangent_slice = read_only_tangent.as_slice().unwrap();
                let tangent_vec = nalgebra::SVector::<f64, $dof>::from_column_slice(tangent_slice);

                let result = <$rust_group>::exp(&tangent_vec);
                Ok(Self { inner: result })
            }

            #[staticmethod]
            fn from_params(params: &PyArray1<f64>) -> Result<Self, PyArray1DimMismatch> {
                check_array1_dim!(params, $params)?;
                let read_only_params = params.readonly();
                let params_slice = read_only_params.as_slice().unwrap();
                let params_vec = nalgebra::SVector::<f64, $params>::from_column_slice(params_slice);

                Ok(Self {
                    inner: <$rust_group>::from_params(&params_vec),
                })
            }

            fn group_mul(&self, other: &$py_group) -> Self {
                Self {
                    inner: self.inner * &other.inner,
                }
            }

            #[staticmethod]
            fn hat(
                omega: &PyArray1<f64>,
                py: Python,
            ) -> Result<Py<PyArray2<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(omega, $dof)?;
                let read_only_omega = omega.readonly();
                let omega_slice = read_only_omega.as_slice().unwrap();
                let omega_vec = nalgebra::SVector::<f64, $dof>::from_column_slice(omega_slice);

                let result = <$rust_group>::hat(&omega_vec);
                Ok(PyArray1::from_slice(py, result.as_slice())
                    .reshape([$ambient, $ambient])
                    .unwrap()
                    .to_owned())
            }

            #[new]
            fn identity() -> Self {
                Self {
                    inner: <$rust_group>::identity(),
                }
            }

            fn inverse(&self) -> Self {
                Self {
                    inner: self.inner.inverse(),
                }
            }

            fn log(&self, py: Python) -> Py<PyArray1<f64>> {
                let log = self.inner.log();
                PyArray1::from_slice(py, log.as_slice()).to_owned()
            }

            fn matrix(&self, py: Python) -> Py<PyArray2<f64>> {
                let matrix = self.inner.matrix();
                PyArray1::from_slice(py, matrix.as_slice())
                    .reshape([$ambient, $ambient])
                    .unwrap()
                    .to_owned()
            }

            fn params(&self, py: Python) -> Py<PyArray1<f64>> {
                let params = self.inner.params();
                PyArray1::from_slice(py, params.as_slice()).to_owned()
            }

            fn set_params(&mut self, params: &PyArray1<f64>) -> Result<(), PyArray1DimMismatch> {
                check_array1_dim!(params, $params)?;
                let read_only_params = params.readonly();
                let params_slice = read_only_params.as_slice().unwrap();
                let params_vec = nalgebra::SVector::<f64, $params>::from_column_slice(params_slice);

                self.inner.set_params(&params_vec);

                Ok(())
            }

            #[staticmethod]
            fn to_ambient(
                point: &PyArray1<f64>,
                py: Python,
            ) -> Result<Py<PyArray1<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(point, $point)?;
                let read_only_point = point.readonly();
                let point_slice = read_only_point.as_slice().unwrap();
                let point_vec = nalgebra::SVector::<f64, $point>::from_column_slice(point_slice);

                let result = <$rust_group>::to_ambient(&point_vec);
                Ok(PyArray1::from_slice(py, result.as_slice()).to_owned())
            }

            fn transform(
                &self,
                py: Python,
                point: &PyArray1<f64>,
            ) -> Result<Py<PyArray1<f64>>, PyArray1DimMismatch> {
                check_array1_dim!(point, $point)?;
                let read_only_point = point.readonly();
                let point_slice = read_only_point.as_slice().unwrap();
                let point_vec = nalgebra::SVector::<f64, $point>::from_column_slice(point_slice);

                let result = self.inner.transform(&point_vec);
                Ok(PyArray1::from_slice(py, result.fixed_rows::<$point>(0).as_slice()).to_owned())
            }

            #[staticmethod]
            fn vee(
                omega_hat: &PyArray2<f64>,
                py: Python,
            ) -> Result<Py<PyArray1<f64>>, PyArray1DimMismatch> {
                let omega_hat = omega_hat.readonly();
                let omega_hat_slice = omega_hat.as_slice().unwrap();
                let omega_hat_mat = nalgebra::SMatrix::<f64, $ambient, $ambient>::from_column_slice(
                    omega_hat_slice,
                );

                let result = <$rust_group>::vee(&omega_hat_mat);
                Ok(PyArray1::from_slice(py, result.as_slice()).to_owned())
            }

            fn __mul__(&self, other: &$py_group) -> Self {
                Self {
                    inner: self.inner * &other.inner,
                }
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
                translation: &PyArray1<f64>,
                rotation: $py_factor_group,
            ) -> Result<Self, PyArray1DimMismatch> {
                check_array1_dim!(translation, $point)?;
                let read_only_translation = translation.readonly();
                let translation_slice = read_only_translation.as_slice().unwrap();
                let translation_vec =
                    nalgebra::SVector::<f64, $point>::from_column_slice(translation_slice);

                Ok(Self {
                    inner: <$rust_group>::from_translation_and_rotation(
                        &translation_vec,
                        &rotation.inner,
                    ),
                })
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

            fn set_translation(
                &mut self,
                translation: &PyArray1<f64>,
            ) -> Result<(), PyArray1DimMismatch> {
                check_array1_dim!(translation, $point)?;

                let read_only_translation = translation.readonly();
                let translation_slice = read_only_translation.as_slice().unwrap();
                let translation_vec =
                    nalgebra::SVector::<f64, $point>::from_column_slice(translation_slice);

                self.inner.set_translation(&translation_vec);

                Ok(())
            }

            fn set_rotation(&mut self, rotation: $py_factor_group) {
                self.inner.set_rotation(&rotation.inner);
            }
        }
    };
}

augment_py_product_group_class!(PyIsometry2, Isometry2<f64>, PyRotation2, 2);
augment_py_product_group_class!(PyIsometry3, Isometry3<f64>, PyRotation3, 3);
