
struct ViewTransform {
    scene_from_camera: mat4x4<f32>,
};

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

struct Frustum {
    width: f32,
    height: f32,
    near: f32,
    far: f32,
    // fx, fy, px, py or just here to debug distortion lut table.
    fx: f32,
    fy: f32,
    px: f32,
    py: f32,
};

struct DistortionLut {
    lut_offset_x: f32,
    lut_offset_y: f32,
    lut_range_x: f32,
    lut_range_y: f32,
};

fn z1_plane_to_distorted(point_in_proj: vec3<f32>, frustum: Frustum, lut: DistortionLut) -> vec3<f32> {
    var width = frustum.width;
    var height = frustum.height;
    var lut_offset_x = lut.lut_offset_x;
    var lut_offset_y = lut.lut_offset_y;
    var lut_range_x = lut.lut_range_x;
    var lut_range_y = lut.lut_range_y;

    let u = clamp((width - 1.0) * (point_in_proj.x -lut_offset_x) / lut_range_x, 0.0, width - 1.00001);
    let v = clamp((height - 1.0) * (point_in_proj.y -lut_offset_y) / lut_range_y, 0.0, height - 1.00001);

    // Manual implementation of bilinear interpolation.
    // This is to workaround apparent limitations of wgpu - such as no/limited support for
    // sampling of f32 textures and sampling in the vertex shader.
    // TDDO: Figure out how to use sampling in vertex shader or maybe undistort in fragment shader
    //       (first render pinhole image to texture, then undistort in fragment shader).
    let u0 = i32(u); // left nearest coordinate
    let v0 = i32(v); // top nearest coordinate
    let u1 = u0 + 1; // right nearest coordinate
    let v1 = v0 + 1; // bottom nearest coordinate
    let frac_u = u - f32(u0); // fractional part of u
    let frac_v = v - f32(v0); // fractional part of v
    var val00 = textureLoad(distortion_texture, vec2<i32>(u0, v0), 0).xy;
    var val01 = textureLoad(distortion_texture, vec2<i32>(u0, v1), 0).xy;
    var val10 = textureLoad(distortion_texture, vec2<i32>(u1, v0), 0).xy;
    var val11 = textureLoad(distortion_texture, vec2<i32>(u1, v1), 0).xy;
    var val0 = mix(val00, val01, frac_v);
    var val1 = mix(val10, val11, frac_v);
    var val = mix(val0, val1, frac_u);
    return vec3<f32>(val.x, val.y, point_in_proj.z);
}

fn scene_point_to_distorted(scene_point: vec3<f32>,
                            view: ViewTransform,
                            frustum: Frustum,
                            lut: DistortionLut) -> vec3<f32> {
    var point_in_proj = scene_point_to_z1_plane(scene_point, view);
    return z1_plane_to_distorted(point_in_proj, frustum, lut);
}

// map point from pixel coordinates (Computer Vision convention) to clip space coordinates (WebGPU convention)
fn pixel_and_z_to_clip(uv_z: vec2<f32>, z: f32, frustum: Frustum) -> vec4<f32> {
    var width = frustum.width;
    var height = frustum.height;
    var near = frustum.near;
    var far = frustum.far;
    var u = uv_z.x;
    var v = uv_z.y;

    return vec4<f32>(2.0 * (u / width - 0.5),
                    -2.0 * (v / height - 0.5),
                    // todo: Check whether the z value is correct.
                    (far * (z - near)) / (z * (far - near)),
                    1.0);
}
