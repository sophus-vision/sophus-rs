// use sophus_lie::prelude::IsSingleScalar;
// use sophus_sensor::camera_enum::perspective_camera::KannalaBrandtCameraF64;
// use sophus_sensor::camera_enum::perspective_camera::UnifiedCamera;
// use sophus_sensor::dyn_camera::DynCameraF64;

// use sophus_core::calculus::dual::DualScalar;
// use sophus_core::calculus::dual::DualVector;
// use sophus_core::calculus::maps::VectorValuedMapFromVector;
// use sophus_core::linalg::VecF64;
// use sophus_lie::Isometry3;
// use sophus_lie::Isometry3F64;
// use sophus_sensor::KannalaBrandtCamera;
// use sophus_sensor::PinholeCamera;

// fn res_fn<Scalar: IsSingleScalar>(
//     intrinscs: KannalaBrandtCamera<Scalar, 1>,
//     unified: UnifiedCamera<Scalar, 1>,
//     uv_in_image: Scalar::Vector<2>,
// ) -> Scalar::Vector<2> {
//     let point_in_cam = unified.cam_unproj(&uv_in_image);
//     uv_in_image - intrinscs.cam_proj(&point_in_cam)
// }

// /// Reprojection term signature
// #[derive(Clone)]
// pub struct CameraFitTermSignature {
//     /// Pixel measurement
//     pub uv_in_image: VecF64<2>,
//     /// camera/intrinsics index
//     pub entity_indices: [usize; 1],
// }

// impl IsTermSignature<1> for CameraFitTermSignature {
//     type Constants = VecF64<2>;

//     fn c_ref(&self) -> &Self::Constants {
//         &self.uv_in_image
//     }

//     fn idx_ref(&self) -> &[usize; 1] {
//         &self.entity_indices
//     }

//     const DOF_TUPLE: [i64; 3] = [8];
// }

// /// Camera fit cost function
// #[derive(Copy, Clone)]
// pub struct CameraFitCostFn {}

// impl IsResidualFn<8, 1, (KannalaBrandtCameraF64), VecF64<2>> for CameraFitCostFn {
//     fn eval(
//         &self,
//         idx: [usize; 1],
//         (intrinsics,): (KannalaBrandtCameraF64,),
//         var_kinds: [VarKind; 1],
//         robust_kernel: Option<robust_kernel::RobustKernel>,
//         uv_in_image: &VecF64<2>,
//     ) -> Term<8, 2> {
//         // calculate residual
//         let residual = res_fn(
//             intrinsics,
//             world_from_camera_pose,
//             point_in_world,
//             *uv_in_image,
//         );

//         // calculate jacobian wrt intrinsics
//         let d0_res_fn = |x: DualVector<4>| -> DualVector<2> {
//             res_fn(
//                 PinholeCamera::<DualScalar, 1>::from_params_and_size(&x, intrinsics.image_size()),
//                 world_from_camera_pose.to_dual_c(),
//                 DualVector::from_real_vector(point_in_world),
//                 DualVector::from_real_vector(*uv_in_image),
//             )
//         };
//         // calculate jacobian wrt world_from_camera_pose
//         let d1_res_fn = |x: DualVector<6>| -> DualVector<2> {
//             res_fn(
//                 PinholeCamera::<DualScalar, 1>::from_params_and_size(
//                     &DualVector::from_real_vector(*intrinsics.params()),
//                     intrinsics.image_size(),
//                 ),
//                 Isometry3::<DualScalar, 1>::exp(&x).group_mul(&world_from_camera_pose.to_dual_c()),
//                 DualVector::from_real_vector(point_in_world),
//                 DualVector::from_real_vector(*uv_in_image),
//             )
//         };
//         // calculate jacobian wrt point_in_world
//         let d2_res_fn = |x: DualVector<3>| -> DualVector<2> {
//             res_fn(
//                 PinholeCamera::<DualScalar, 1>::from_params_and_size(
//                     &DualVector::from_real_vector(*intrinsics.params()),
//                     intrinsics.image_size(),
//                 ),
//                 world_from_camera_pose.to_dual_c(),
//                 x,
//                 DualVector::from_real_vector(*uv_in_image),
//             )
//         };

//         (
//             || {
//                 VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
//                     d0_res_fn,
//                     *intrinsics.params(),
//                 )
//             },
//             || {
//                 VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
//                     d1_res_fn,
//                     VecF64::<6>::zeros(),
//                 )
//             },
//             || {
//                 VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
//                     d2_res_fn,
//                     point_in_world,
//                 )
//             },
//         )
//             .make_term(idx, var_kinds, residual, robust_kernel, None)
//     }
// }

// pub fn fit_camera(camera: &mut DynCameraF64) -> UnifiedCameraF64 {
//     let intrinsics = camera.intrinsics();
//     let image_size = intrinsics.image_size();
//     let mut fit_camera = UnifiedCameraF64::new(image_size);
//     fit_camera.set_intrinsics(intrinsics);
//     fit_camera
// }
