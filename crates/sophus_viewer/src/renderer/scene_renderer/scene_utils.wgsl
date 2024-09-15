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
    alpha: f32,
    beta: f32,
};

struct Zoom2d {
    translation_x: f32,
    translation_y: f32,
    scaling_x: f32,
    scaling_y: f32,
};

struct ViewTransform {
    scene_from_camera: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> frustum_uniforms: Frustum;

@group(0) @binding(1)
var<uniform> zoom: Zoom2d;

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
fn z1_plane_to_distorted(point_in_proj: vec3<f32>, frustum: Frustum) -> vec3<f32> {
    var u = 0.0;
    var v = 0.0;
    if (frustum.alpha == 0.0) {
        u = point_in_proj.x * frustum.fx + frustum.px;
        v = point_in_proj.y * frustum.fy + frustum.py;
    } else {
        let r2 = point_in_proj.x * point_in_proj.x + point_in_proj.y * point_in_proj.y;
        let rho2 = frustum.beta * r2 + 1.0;
        let rho = sqrt(rho2);
        let norm = frustum.alpha * rho + (1.0 - frustum.alpha);
        u = point_in_proj.x / norm * frustum.fx + frustum.px;
        v = point_in_proj.y / norm * frustum.fy + frustum.py;
    }
    return vec3<f32>(u, v, point_in_proj.z);
}

fn scene_point_to_distorted(scene_point: vec3<f32>,
                            view: ViewTransform,
                            frustum: Frustum) -> vec3<f32> {
    var point_in_proj = scene_point_to_z1_plane(scene_point, view);
    return z1_plane_to_distorted(point_in_proj, frustum);
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
