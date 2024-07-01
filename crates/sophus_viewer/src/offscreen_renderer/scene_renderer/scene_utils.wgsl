struct Frustum {
    camera_image_width: f32, // <= NOT the viewport width
    camera_image_height: f32, // <= NOT the viewport height
    near: f32,
    far: f32,
    // fx, fy, px, py or just here to debug distortion lut table.
    fx: f32,
    fy: f32,
    px: f32,
    py: f32,
};

struct Zoom2d {
    translation_x: f32,
    translation_y: f32,
    scaling_x: f32,
    scaling_y: f32,
};

struct DistortionLut {
    lut_offset_x: f32,
    lut_offset_y: f32,
    lut_range_x: f32,
    lut_range_y: f32,
};

struct ViewTransform {
    scene_from_camera: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> frustum_uniforms: Frustum;

@group(0) @binding(1)
var<uniform> zoom: Zoom2d;

@group(0) @binding(2)
var<uniform> lut_uniform: DistortionLut;

@group(0) @binding(3)
var<uniform> view_uniform: ViewTransform;


fn scene_point_to_z1_plane(scene_point: vec3<f32>,
                                 view: ViewTransform) -> vec3<f32> {
    var scene_from_camera = view.scene_from_camera;

    // map point from scene to camera frame
    var hpoint_in_cam = scene_from_camera * vec4<f32>(scene_point, 1.0);

    // perspective point in camera frame
    var point_in_cam = hpoint_in_cam.xyz / hpoint_in_cam.w;
    var z = point_in_cam.z;
    // point projected to the z=1 plane
    var point_in_proj = point_in_cam.xy/point_in_cam.z;

    return vec3<f32>(point_in_proj.x, point_in_proj.y, z);
}
fn z1_plane_to_distorted(point_in_proj: vec3<f32>, frustum: Frustum, lut: DistortionLut) -> vec3<f32> {
    var width = frustum.camera_image_width;
    var height = frustum.camera_image_height;
    var lut_offset_x = lut.lut_offset_x;
    var lut_offset_y = lut.lut_offset_y;
    var lut_range_x = lut.lut_range_x;
    var lut_range_y = lut.lut_range_y;

    let x_lut =
        clamp(
            (width -1.0 ) * (point_in_proj.x -lut_offset_x) / lut_range_x,
            0.0,
            width - 1.00001
        );
    let y_lut =
        clamp(
            (height -1.0 ) * (point_in_proj.y -lut_offset_y) / lut_range_y,
            0.0,
            height - 1.00001
        );

    // Manual implementation of bilinear interpolation.
    // This is to workaround apparent limitations of wgpu - such as no/limited support for
    // sampling of f32 textures and sampling in the vertex shader.
    // TDDO: Figure out how to use sampling in vertex shader or maybe undistort in fragment shader
    //       (first render pinhole image to texture, then undistort in fragment shader).
    let x0 = i32(x_lut); // left nearest coordinate
    let y0 = i32(y_lut); // top nearest coordinate
    let x1 = x0 + 1; // right nearest coordinate
    let y1 = y0 + 1; // bottom nearest coordinate
    let frac_x = x_lut - f32(x0); // fractional part of u
    let frac_y = y_lut - f32(y0); // fractional part of v
    var val00 = textureLoad(distortion_texture, vec2<i32>(x0, y0), 0).xy;
    var val01 = textureLoad(distortion_texture, vec2<i32>(x0, y1), 0).xy;
    var val10 = textureLoad(distortion_texture, vec2<i32>(x1, y0), 0).xy;
    var val11 = textureLoad(distortion_texture, vec2<i32>(x1, y1), 0).xy;
    var val0 = mix(val00, val01, frac_y);
    var val1 = mix(val10, val11, frac_y);
    var val = mix(val0, val1, frac_x);
    let u = val.x;
    let v = val.y;

    // to debug distortion lut table - just using pinhole projection
    // let u = point_in_proj.x * frustum.fx + frustum.px;
    // let v = point_in_proj.y * frustum.fy + frustum.py;
    return vec3<f32>(u, v, point_in_proj.z);
}

fn scene_point_to_distorted(scene_point: vec3<f32>,
                            view: ViewTransform,
                            frustum: Frustum,
                            lut: DistortionLut) -> vec3<f32> {
    var point_in_proj = scene_point_to_z1_plane(scene_point, view);
    return z1_plane_to_distorted(point_in_proj, frustum, lut);
}

// map point from pixel coordinates (Computer Vision convention) to clip space coordinates (WebGPU convention)
fn pixel_and_z_to_clip(uv_z: vec2<f32>, z: f32, frustum: Frustum, zoom: Zoom2d) -> vec4<f32> {
    var width = frustum.camera_image_width;
    var height = frustum.camera_image_height;
    var near = frustum.near;
    var far = frustum.far;
    var u = uv_z.x * zoom.scaling_x + zoom.translation_x;
    var v = uv_z.y * zoom.scaling_y + zoom.translation_y;

    let z_clip = (far * (z - near)) / (z * (far - near));

    return vec4<f32>(2.0 * ((u + 0.5) / width - 0.5),
                    -2.0 * ((v + 0.5) / height - 0.5),
                    z_clip,
                    1.0);
}
