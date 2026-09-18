@group(0) @binding(0)
var<uniform> camera: CameraProperties;

@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

@group(0) @binding(2)
var<uniform> ortho_camera: PinholeModel;

@group(0) @binding(3)
var<uniform> view_uniform: CameraPose;

// what the scene left at each pixel, as inverse distance along the ray - zero where it left nothing
@group(1) @binding(0)
var scene_inverse_depth: texture_2d<f32>;

struct VertexOut {
    @location(0) color: vec4<f32>,
    // offset from the centre of the point, and its half extent - both in image pixels, so that
    // the fragment shader can work out how much of the pixel the point actually covers
    @location(1) offset: vec2<f32>,
    @location(2) radius: f32,
    @location(3) feather: f32,
    // inverse distance of the point itself, which is what it is occluded by
    @location(4) inverse_distance: f32,
    @builtin(position) position: vec4<f32>,
};

// What the scene left behind is written as well as read: these belong to the scene, and the point
// under the pointer needs a distance for anything to be picked off it.
struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) inverse_distance: f32,
};

@vertex
fn vs_main(
     @location(0) position: vec3<f32>,
     @location(1) point_size: f32,
     @location(2) color: vec4<f32>,
     @builtin(vertex_index) idx: u32) -> VertexOut
{
    var out: VertexOut;
    out.color = color;

    let in_camera = (view_uniform.camera_from_entity * vec4<f32>(position, 1.0)).xyz;
    // Behind the camera, or so far off axis that it has no place on the z = 1 plane at all. There
    // is nowhere right to put it, so it is put outside clip space and never rasterized.
    if (in_camera.z <= 1e-6) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    // one view-port pixel, in image pixels - the width of the antialiased edge
    let feather = ortho_camera.viewport_scale;
    let radius = 0.5 * point_size * feather;
    // the quad is grown by the feather, so that the fragments which are only partly covered exist
    let outer = radius + feather;

    // Where the point lands in the *distorted* image: the camera model's own projection, not the
    // undistorted intermediate's. A point has no extent to bend, so this is exact.
    //
    // The 2d zoom is already folded into that camera model - it is an affine step on the image
    // plane, so it goes into the intrinsics exactly - which is why it is not applied again here.
    // A 2d renderable has to apply it itself, since it is anchored in image pixels rather than
    // projected through anything.
    let anchor = z1_plane_to_distorted(in_camera.xy / in_camera.z, camera);

    let mod6 = idx % 6u;
    var offset = vec2<f32>(-outer, -outer);
    if mod6 == 1u || mod6 == 3u {
        offset = vec2<f32>(outer, -outer);
    } else if mod6 == 2u || mod6 == 4u {
        offset = vec2<f32>(-outer, outer);
    } else if mod6 == 5u {
        offset = vec2<f32>(outer, outer);
    }

    out.position = ortho_pixel_to_clip(anchor + offset,
        vec2<f32>(camera.camera_image_width, camera.camera_image_height));
    out.offset = offset;
    out.radius = radius;
    out.feather = feather;
    out.inverse_distance = 1.0 / length(in_camera);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    if (occluded(scene_inverse_depth, in.position.xy, in.inverse_distance)) {
        discard;
    }
    let inside = in.radius - length(in.offset);
    let coverage = clamp(inside / in.feather + 0.5, 0.0, 1.0);
    // A fragment which covers nothing still writes its depth, so the ones outside the shape are
    // dropped rather than left to claim the pixel.
    if (coverage <= 0.0) {
        discard;
    }

    var out: FragmentOut;
    out.color = vec4<f32>(in.color.rgb, in.color.a * coverage);
    out.inverse_distance = in.inverse_distance;
    return out;
}
