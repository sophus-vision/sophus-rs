struct OrthoCamera {
     width: f32,
     height: f32,
     viewport_scale: f32,
     dummy: f32,
};

struct Zoom2d {
    translation_x: f32,
    translation_y: f32,
    scaling_x: f32,
    scaling_y: f32,
};

@group(0) @binding(0)
var<uniform> ortho_camera: OrthoCamera;

@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

// apply zoom and convert from pixel to clip space
fn pixel_and_z_to_clip(uv: vec2<f32>, zoom_2d: Zoom2d, ortho_camera: OrthoCamera) -> vec4<f32> {
    var p_x = uv.x * zoom_2d.scaling_x + zoom_2d.translation_x;
    var p_y = uv.y * zoom_2d.scaling_y + zoom_2d.translation_y;

    return vec4<f32>(2.0 * (p_x + 0.5) / ortho_camera.width - 1.0,
                     2.0 - 2.0 * (p_y + 0.5) / ortho_camera.height - 1.0,
                     0.0,
                     1.0);
}
