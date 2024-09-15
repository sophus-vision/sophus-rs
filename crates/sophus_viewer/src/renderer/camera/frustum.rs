#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Frustum {
    pub(crate) camera_image_width: f32, // <= this is NOT the view-port width
    pub(crate) camera_image_height: f32, // <= this is NOT the view-port height
    pub(crate) near: f32,
    pub(crate) far: f32,
    // pinhole parameter
    pub(crate) fx: f32,
    pub(crate) fy: f32,
    pub(crate) px: f32,
    pub(crate) py: f32,
    //
    pub(crate) alpha: f32, // if alpha == 0, then we use the pinhole model
    pub(crate) beta: f32,
}
