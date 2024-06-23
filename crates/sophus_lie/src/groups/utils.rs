use sophus_core::linalg::bool_mask::IsBoolMask;
use sophus_core::linalg::matrix::IsMatrix;
use sophus_core::linalg::scalar::IsScalar;

// template <class TScalar, int kMatrixDim>
// auto calcW(
//     Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> const &omega,
//     TScalar const theta,
//     TScalar const sigma) -> Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> {
//   using std::abs;
//   using std::cos;
//   using std::exp;
//   using std::sin;
//   static Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> const kI =
//       Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim>::Identity();
//   static TScalar const kOne(1);
//   static TScalar const kHalf(0.5);
//   Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> const omega2 = omega * omega;
//   TScalar const scale = exp(sigma);
//   TScalar a;

//   TScalar b;

//   TScalar c;
//   if (abs(sigma) < kEpsilon<TScalar>) {
//     c = kOne;
//     if (abs(theta) < kEpsilon<TScalar>) {
//       a = kHalf;
//       b = TScalar(1. / 6.);
//     } else {
//       TScalar theta_sq = theta * theta;
//       a = (kOne - cos(theta)) / theta_sq;
//       b = (theta - sin(theta)) / (theta_sq * theta);
//     }
//   } else {
//     c = (scale - kOne) / sigma;
//     if (abs(theta) < kEpsilon<TScalar>) {
//       TScalar sigma_sq = sigma * sigma;
//       a = ((sigma - kOne) * scale + kOne) / sigma_sq;
//       b = (scale * kHalf * sigma_sq + scale - kOne - sigma * scale) /
//           (sigma_sq * sigma);
//     } else {
//       TScalar theta_sq = theta * theta;
//       TScalar tmp_a = scale * sin(theta);
//       TScalar tmp_b = scale * cos(theta);
//       TScalar tmp_c = theta_sq + sigma * sigma;
//       a = (tmp_a * sigma + (kOne - tmp_b) * theta) / (theta * tmp_c);
//       b = (c - ((tmp_b - kOne) * sigma + tmp_a * theta) / (tmp_c)) * kOne /
//           (theta_sq);
//     }
//   }
//   return a * omega + b * omega2 + c * kI;
// }

pub(crate) fn calc_mat_v<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize, const DIM: usize>(
    omega: &S::Matrix<DIM, DIM>,
    theta: &S,
    sigma: &S,
) -> S::Matrix<DIM, DIM> {
    let scale = sigma.clone().exp();
    let omaga_sq: S::Matrix<DIM, DIM> = omega.clone().mat_mul(omega.clone());
    let theta_sq = theta.clone() * theta.clone();
    let sigma_sq = sigma.clone() * sigma.clone();

    let sin_theta = theta.clone().sin();
    let cos_theta = theta.clone().cos();

    let s_sin_theta = scale.clone() * sin_theta.clone();
    let s_cos_theta = scale.clone() * cos_theta.clone();
    let theta_sq_plus_sigma_sq = theta_sq.clone() + sigma_sq.clone();

    let eps = S::from_f64(1e-5);

    // 1. Check if theta is near zero
    let theta_near_zero = theta.clone().abs().less_equal(&eps);

    // for c

    //     c = kOne;
    let c_s0 = S::from_f64(1.0);
    //     c = (scale - kOne) / sigma;
    let c_s = (scale.clone() - S::from_f64(1.0)) / sigma.clone();

    // for a
    let a_s0 = {
        //a = kHalf;
        let a_s0_t0 = S::from_f64(0.5);
        //  a = (kOne - cos(theta)) / theta_sq;
        let a_s0_t = (S::from_f64(1.0) - cos_theta.clone()) / theta_sq.clone();
        a_s0_t0.select(&theta_near_zero, a_s0_t)
    };
    let a_s = {
        // a = ((sigma - kOne) * scale + kOne) / sigma_sq;
        let a_s_t0 = ((sigma.clone() - S::from_f64(1.0)) * scale.clone() + S::from_f64(1.0))
            / (sigma_sq.clone());
        // a = (tmp_a * sigma + (kOne - tmp_b) * theta) / (theta * tmp_c);
        let a_s_t = (s_sin_theta.clone() * sigma.clone()
            + (S::from_f64(1.0) - s_cos_theta.clone()) * theta.clone())
            / (theta.clone() * theta_sq_plus_sigma_sq.clone());
        a_s_t0.select(&theta_near_zero, a_s_t)
    };

    // for b
    let b_s0 = {
        //       b = TScalar(1. / 6.);
        let b_s0_t0 = S::from_f64(1.0 / 6.0);
        //       b = (theta - sin(theta)) / (theta_sq * theta);
        let b_s0_t = (theta.clone() - sin_theta.clone()) / (theta.clone() * theta_sq.clone());
        b_s0_t0.select(&theta_near_zero, b_s0_t)
    };
    let b_s = {
        //       b = (scale * kHalf * sigma_sq + scale - kOne - sigma * scale) /
        //           (sigma_sq * sigma);
        let b_s_t0 = (scale.clone() * S::from_f64(0.5) * sigma_sq.clone() + scale.clone()
            - S::from_f64(1.0)
            - sigma.clone() * scale.clone())
            / (sigma_sq.clone() * sigma.clone());
   //       b = (c - ((tmp_b - kOne) * sigma + tmp_a * theta) / (tmp_c)) * kOne /
//           (theta_sq);
        let b_s_t = (c_s.clone()
            - ((s_cos_theta.clone() - S::from_f64(1.0)) * sigma.clone()
                + s_sin_theta.clone() * theta.clone())
                / theta_sq_plus_sigma_sq.clone())
            * S::from_f64(1.0)
            / theta_sq.clone();
        b_s_t0.select(&theta_near_zero, b_s_t)
    };

    // 2. Check if sigma is near zero
    let sigma_near_zero = sigma.clone().abs().less_equal(&eps);
    let a = a_s0.select(&sigma_near_zero, a_s);
    let b = b_s0.select(&sigma_near_zero, b_s);
    let c = c_s0.select(&sigma_near_zero, c_s.clone());

    println!("-a: {:?}", a);
    println!("-b: {:?}", b);
    println!("-c: {:?}", c);

    omega.scaled(a) + omaga_sq.scaled(b) + S::Matrix::<DIM, DIM>::identity().scaled(c)
}

// template <class TScalar, int kMatrixDim>
// auto calcWInv(
//     Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> const &omega,
//     TScalar const theta,
//     TScalar const sigma,
//     TScalar const scale) -> Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> {
//   using std::abs;
//   using std::cos;
//   using std::sin;
//   static Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> const kI =
//       Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim>::Identity();
//   static TScalar const kHalf(0.5);
//   static TScalar const kOne(1);
//   static TScalar const kTwo(2);
//   Eigen::Matrix<TScalar, kMatrixDim, kMatrixDim> const omega2 = omega * omega;
//   TScalar const scale_sq = scale * scale;
//   TScalar const theta_sq = theta * theta;
//   TScalar const sin_theta = sin(theta);
//   TScalar const cos_theta = cos(theta);

//   TScalar a;

//   TScalar b;

//   TScalar c;
//   if (abs(sigma * sigma) < kEpsilon<TScalar>) {
//     c = kOne - kHalf * sigma;
//     a = -kHalf;
//     if (abs(theta_sq) < kEpsilon<TScalar>) {
//       b = TScalar(1. / 12.);
//     } else {
//       b = (theta * sin_theta + kTwo * cos_theta - kTwo) /
//           (kTwo * theta_sq * (cos_theta - kOne));
//     }
//   } else {
//     TScalar const scale_cu = scale_sq * scale;
//     c = sigma / (scale - kOne);
//     if (abs(theta_sq) < kEpsilon<TScalar>) {
//       a = (-sigma * scale + scale - kOne) / ((scale - kOne) * (scale - kOne));
//       b = (scale_sq * sigma - kTwo * scale_sq + scale * sigma + kTwo * scale) /
//           (kTwo * scale_cu - TScalar(6) * scale_sq + TScalar(6) * scale - kTwo);
//     } else {
//       TScalar const s_sin_theta = scale * sin_theta;
//       TScalar const s_cos_theta = scale * cos_theta;
//       a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
//           (theta * (scale_sq - kTwo * s_cos_theta + kOne));
//       b = -scale *
//           (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
//            scale * sigma + sigma * cos_theta - sigma) /
//           (theta_sq * (scale_cu - kTwo * scale * s_cos_theta - scale_sq +
//                        kTwo * s_cos_theta + scale - kOne));
//     }
//   }
//   return a * omega + b * omega2 + c * kI;
// }

pub(crate) fn calc_mat_w_inv<S: IsScalar<BATCH_SIZE>, const BATCH_SIZE: usize, const DIM: usize>(
    omega: &S::Matrix<DIM, DIM>,
    theta: &S,
    sigma: &S,
    nrm: &S,
) -> S::Matrix<DIM, DIM> {
    let nrm_sq = nrm.clone() * nrm.clone();
    let theta_sq = theta.clone() * theta.clone();
    let sin_theta = theta.clone().sin();
    let cos_theta = theta.clone().cos();

    // 1. Check if theta is near zero
    let eps = S::from_f64(1e-5);
    let theta_near_zero = theta.clone().abs().less_equal(&eps);

    // for a

    //     a = -kHalf;
    let a_s0 = S::from_f64(-0.5);
    let a_s: S = {
        //       a = (-sigma * nrm + nrm - kOne) / ((nrm - kOne) * (nrm - kOne));
        let a_s_t0 = (-sigma.clone() * nrm.clone() + nrm.clone() - S::from_f64(1.0))
            / ((nrm.clone() - S::from_f64(1.0)) * (nrm.clone() - S::from_f64(1.0)));
        //       a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
        //           (theta * (nrm_sq - kTwo * s_cos_theta + kOne));
        let a_s_t = (theta.clone() * nrm.clone() * cos_theta.clone()
            - theta.clone()
            - sigma.clone() * nrm.clone() * sin_theta.clone())
            / (theta.clone()
                * (nrm_sq.clone() - S::from_f64(2.0) * nrm.clone() * cos_theta.clone()
                    + S::from_f64(1.0)));
        a_s_t0.select(&theta_near_zero, a_s_t)
    };

    // for b
    let b_s0 = {
        //       b = TScalar(1. / 12.);
        let b_s0_t0 = S::from_f64(1.0 / 12.0);
        //       b = (theta * sin_theta + kTwo * cos_theta - kTwo) /
        //           (kTwo * theta_sq * (cos_theta - kOne));
        let b_s0_t = (theta.clone() * sin_theta.clone() + S::from_f64(2.0) * cos_theta.clone()
            - S::from_f64(2.0))
            / (S::from_f64(2.0) * theta_sq.clone() * (cos_theta.clone() - S::from_f64(1.0)));
        b_s0_t0.select(&theta_near_zero, b_s0_t)
    };

    let b_s = {
        //       b = (nrm_sq * sigma - kTwo * nrm_sq + nrm * sigma + kTwo * nrm) /
        //           (kTwo * nrm_cu - TScalar(6) * nrm_sq + TScalar(6) * nrm - kTwo);
        let b_s_t0 = (nrm_sq.clone() * sigma.clone() - S::from_f64(2.0) * nrm_sq.clone()
            + nrm.clone() * sigma.clone()
            + S::from_f64(2.0) * nrm.clone())
            / (S::from_f64(2.0) * nrm_sq.clone() * nrm.clone()
                - S::from_f64(6.0) * nrm_sq.clone()
                + S::from_f64(6.0) * nrm.clone()
                - S::from_f64(2.0));
        //       b = -nrm *
        //           (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
        //            nrm * sigma + sigma * cos_theta - sigma) /
        //           (theta_sq * (nrm_cu - kTwo * nrm * s_cos_theta - nrm_sq +
        //                        kTwo * s_cos_theta + nrm - kOne));
        let b_s_t = (-nrm.clone()
            * (theta.clone() * sin_theta.clone() - theta.clone().sin()
                + sigma.clone() * nrm.clone() * cos_theta.clone()
                - nrm.clone() * sigma.clone()
                + sigma.clone() * cos_theta.clone()
                - sigma.clone()))
            / (theta_sq.clone()
                * (nrm_sq.clone() * nrm.clone()
                    - S::from_f64(2.0) * nrm.clone() * nrm.clone() * cos_theta.clone()
                    - nrm_sq.clone()
                    + S::from_f64(2.0) * nrm.clone() * cos_theta.clone()
                    + nrm.clone()
                    - S::from_f64(1.0)));
        b_s_t0.select(&theta_near_zero, b_s_t)
    };

    // for c

    //     c = kOne - kHalf * sigma;
    let c_s0 = S::from_f64(1.0) - S::from_f64(0.5) * sigma.clone();
    //     c = sigma / (nrm - kOne);
    let c_s = sigma.clone() / (nrm.clone() - S::from_f64(1.0));

    // 2. Check if sigma is near zero
    let sigma_near_zero = sigma.clone().abs().less_equal(&eps);

    // println!("a_s0: {:?}", a_s0());
    // println!("a_s: {:?}", a_s());

    let a = a_s0.select(&sigma_near_zero, a_s);
    let b = b_s0.select(&sigma_near_zero, b_s);
    let c = c_s0.select(&sigma_near_zero, c_s);

    // println!("sigma: {:?}", sigma);
    // println!("nrm: {:?}", nrm);
    // println!("theta: {:?}", theta);
    println!("a: {:?}", a);
    println!("b: {:?}", b);
    println!("c: {:?}", c);

    omega.scaled(a)
        + omega.clone().mat_mul(omega.clone()).scaled(b)
        + S::Matrix::<DIM, DIM>::identity().scaled(c)
}
