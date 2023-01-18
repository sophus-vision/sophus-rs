#[cxx::bridge(namespace = "sophus")]
pub mod ffi {

    #[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
    pub struct FfiImageSize {
        width: usize,
        height: usize,
    }

    #[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
    pub struct FfiImageShape {
        size: FfiImageSize,
        pitch_in_bytes: usize,
    }

    #[derive(Debug, Clone)]
    pub struct FfiIntensityImage {
        layout: FfiImageShape,
        pixel_format: FfiRuntimePixelType,
        data: SharedPtr<u8>,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct FfiRuntimePixelType {
        is_floating_point: bool, // unsigned otherwise
        num_channels: usize,
        num_bytes_per_pixel_channel: usize,
    }

    unsafe extern "C++" {
        include!("sophus-rs/include/sophus_wrapper.h");

        type FfiMutIntensityImage;

        fn create_mut_intensity_image_from_size(
            size: FfiImageSize,
            t: FfiRuntimePixelType,
        ) -> UniquePtr<FfiMutIntensityImage>;

        fn create_intensity_image_from_mut(
            mut_image: &mut UniquePtr<FfiMutIntensityImage>,
        ) -> FfiIntensityImage;

        fn get_raw_ptr(img: &FfiIntensityImage) -> *const u8;
        fn get_mut_raw_ptr(img: &UniquePtr<FfiMutIntensityImage>) -> *mut u8;

        fn has_u8_img(img: &FfiIntensityImage) -> bool;
        fn has_u16_img(img: &FfiIntensityImage) -> bool;
        fn has_f32_img(img: &FfiIntensityImage) -> bool;
        fn has_3u8_img(img: &FfiIntensityImage) -> bool;
        fn has_3u16_img(img: &FfiIntensityImage) -> bool;
        fn has_3f32_img(img: &FfiIntensityImage) -> bool;
        fn has_4u8_img(img: &FfiIntensityImage) -> bool;
        fn has_4u16_img(img: &FfiIntensityImage) -> bool;
        fn has_4f32_img(img: &FfiIntensityImage) -> bool;

    }
}
