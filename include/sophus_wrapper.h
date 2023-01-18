#pragma once

#include "rust/cxx.h"

#include <sophus/image/runtime_image_types.h>

namespace sophus {

struct FfiIntensityImage;
struct FfiImageShape;
struct FfiImageSize;
struct FfiRuntimePixelType;

struct FfiMutIntensityImage {
  FfiMutIntensityImage() = default;

  FfiMutIntensityImage(
      MutRuntimeImage<IntensityImagePredicate>&& mut_runtime_image)
      : mut_runtime_image(std::move(mut_runtime_image)) {}

  MutRuntimeImage<IntensityImagePredicate> mut_runtime_image;
};

std::unique_ptr<FfiMutIntensityImage> create_mut_intensity_image_from_size(
    FfiImageSize s, FfiRuntimePixelType t);

FfiIntensityImage create_intensity_image_from_mut(
    std::unique_ptr<FfiMutIntensityImage>& mut_image);

uint8_t const* get_raw_ptr(FfiIntensityImage const& img);
uint8_t* get_mut_raw_ptr(std::unique_ptr<FfiMutIntensityImage> const& img);

bool has_u8_img(FfiIntensityImage const& img);
bool has_u16_img(FfiIntensityImage const& img);
bool has_f32_img(FfiIntensityImage const& img);
bool has_3u8_img(FfiIntensityImage const& img);
bool has_3u16_img(FfiIntensityImage const& img);
bool has_3f32_img(FfiIntensityImage const& img);
bool has_4u8_img(FfiIntensityImage const& img);
bool has_4u16_img(FfiIntensityImage const& img);
bool has_4f32_img(FfiIntensityImage const& img);

}  // namespace sophus
