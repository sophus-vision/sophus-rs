/// general camera - either perspective or orthographic
pub mod general_camera;
pub use crate::camera_enum::general_camera::GeneralCameraEnum;

/// perspective camera
pub mod perspective_camera;
pub use crate::camera_enum::perspective_camera::PerspectiveCameraEnum;
