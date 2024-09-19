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




fn scene_point_to_z1_plane_and_depth(scene_point: vec3<f32>,
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

fn z1_plane_to_undistorted(point_in_z1: vec2<f32>, frustum: Frustum) -> vec2<f32> {
    var u = point_in_z1.x * (frustum.fx *0.6)+ frustum.px;
    var v = point_in_z1.y * (frustum.fy *0.6) + frustum.py;
    return vec2<f32>(u, v);
}

fn z1_plane_to_distorted(point_in_z1: vec2<f32>, frustum: Frustum) -> vec2<f32> {
    let fx = frustum.fx;
    let fy = frustum.fy;
    let px = frustum.px;
    let py = frustum.py;
    var u = point_in_z1.x;
    var v = point_in_z1.y;
    let alpha = frustum.alpha;
    let beta = frustum.beta;
    let r2 = u*u + v*v;
    let rho2 = beta * r2 + 1.0;
    let rho = sqrt(rho2);

    let norm = alpha * rho + (1.0 - alpha);

    let mx = u / norm;
    let my = v / norm;

    return vec2<f32>(fx * mx + px, fy * my + py);
}

struct Projection {
    point_in_z1: vec2<f32>,
    uv_undistorted: vec2<f32>,
    uv: vec2<f32>,
    z: f32,
};

fn project_point(point: vec3<f32>, view: ViewTransform, frustum: Frustum, zoom: Zoom2d) -> Projection {
   var out: Projection;
   out.point_in_z1 = scene_point_to_z1_plane_and_depth(point, view).xy;
   out.uv_undistorted = z1_plane_to_undistorted(out.point_in_z1, frustum);
   out.uv = z1_plane_to_distorted(out.point_in_z1, frustum);
   out.z = scene_point_to_z1_plane_and_depth(point, view).z;
   return out;
}

fn distorted_to_z1(uv_distorted: vec2<f32>, frustum: Frustum) -> vec2<f32> {
  

    let u = (uv_distorted.x-frustum.px)/frustum.fx;
    let v = (uv_distorted.y-frustum.py)/frustum.fy;

     
    let r2 = u*u + v*v;
    let gamma = 1.0 - frustum.alpha;

    let nominator = 1.0 - frustum.alpha * frustum.alpha * frustum.beta * r2;
    let denominator = frustum.alpha * sqrt(1.0 - (frustum.alpha - gamma) * frustum.beta * r2) + gamma;

    let k = nominator / denominator;

    return vec2<f32>(u / k, v / k);
}

fn distorted_to_undistorted(uv_distorted: vec2<f32>, frustum: Frustum) -> vec2<f32> {
    let z1 = distorted_to_z1(uv_distorted, frustum);
    return z1_plane_to_undistorted(z1, frustum).xy;
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
