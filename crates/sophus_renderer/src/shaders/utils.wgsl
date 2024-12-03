struct CameraProperties {
    camera_image_width: f32, // <= NOT the viewport width
    camera_image_height: f32, // <= NOT the viewport height
    near: f32,
    far: f32,
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

struct CameraPose {
    camera_from_entity: mat4x4<f32>,
};

struct PinholeModel {
     width: f32,
     height: f32,
     fx: f32,
     fy: f32,
     px: f32,
     py: f32,
     viewport_scale: f32,
     dummy: f32,
};

fn scene_point_to_z1_plane_and_depth(
    scene_point: vec3<f32>,
    view: CameraPose) -> vec3<f32>
{
    var camera_from_entity = view.camera_from_entity;

    // map point from scene to camera frame
    var hpoint_in_cam = camera_from_entity * vec4<f32>(scene_point, 1.0);

    // perspective point in camera frame
    var point_in_cam = hpoint_in_cam.xyz / hpoint_in_cam.w;
    var z = point_in_cam.z;
    // point projected to the z=1 plane
    var point_in_proj = point_in_cam.xy/point_in_cam.z;

    return vec3<f32>(point_in_proj.x, point_in_proj.y, z);
}

fn z1_plane_to_undistorted(point_in_z1: vec2<f32>, pinhole: PinholeModel) -> vec2<f32> {
    var u = point_in_z1.x * pinhole.fx + pinhole.px;
    var v = point_in_z1.y * pinhole.fy + pinhole.py;
    return vec2<f32>(u, v);
}

fn z1_plane_to_distorted(point_in_z1: vec2<f32>, camera: CameraProperties) -> vec2<f32> {
    let fx = camera.fx;
    let fy = camera.fy;
    let px = camera.px;
    let py = camera.py;
    var u = point_in_z1.x;
    var v = point_in_z1.y;
    let alpha = camera.alpha;
    let beta = camera.beta;
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

fn project_point(
    point: vec3<f32>, view: CameraPose,
    pinhole: PinholeModel,
    camera: CameraProperties,
    zoom: Zoom2d) -> Projection
{
   var out: Projection;
   out.point_in_z1 = scene_point_to_z1_plane_and_depth(point, view).xy;
   out.uv_undistorted = z1_plane_to_undistorted(out.point_in_z1, pinhole);
   out.uv = z1_plane_to_distorted(out.point_in_z1, camera);
   out.z = scene_point_to_z1_plane_and_depth(point, view).z;
   return out;
}

fn distorted_to_z1(uv_distorted: vec2<f32>, camera: CameraProperties) -> vec2<f32> {
    let u = (uv_distorted.x-camera.px)/camera.fx;
    let v = (uv_distorted.y-camera.py)/camera.fy;

    let r2 = u*u + v*v;
    let gamma = 1.0 - camera.alpha;

    let nominator = 1.0 - camera.alpha * camera.alpha * camera.beta * r2;
    let denominator = camera.alpha * sqrt(1.0 - (camera.alpha - gamma) * camera.beta * r2) + gamma;

    let k = nominator / denominator;

    return vec2<f32>(u / k, v / k);
}

fn undistort(uv_distorted: vec2<f32>, pinhole: PinholeModel, camera: CameraProperties) -> vec2<f32> {
    let z1 = distorted_to_z1(uv_distorted, camera);
    return z1_plane_to_undistorted(z1, pinhole).xy;
}

// apply zoom and convert from pixel to clip space
fn ortho_pixel_and_z_to_clip(uv: vec2<f32>, zoom_2d: Zoom2d, ortho_camera: PinholeModel) -> vec4<f32> {
    var p_x = uv.x * zoom_2d.scaling_x + zoom_2d.translation_x;
    var p_y = uv.y * zoom_2d.scaling_y + zoom_2d.translation_y;

    return vec4<f32>(2.0 * (p_x + 0.5) / ortho_camera.width - 1.0,
                     2.0 - 2.0 * (p_y + 0.5) / ortho_camera.height - 1.0,
                     0.0,
                     1.0);
}

// map point from pixel coordinates (Computer Vision convention) to clip space coordinates (WebGPU convention)
fn pixel_and_z_to_clip(uv_z: vec2<f32>, z: f32, camera: CameraProperties, zoom: Zoom2d) -> vec4<f32> {
    var width = camera.camera_image_width;
    var height = camera.camera_image_height;
    var near = camera.near;
    var far = camera.far;
    var u = uv_z.x * zoom.scaling_x + zoom.translation_x;
    var v = uv_z.y * zoom.scaling_y + zoom.translation_y;

    if (z < near) {
        return vec4<f32>(2.0 * ((u + 0.5) / width - 0.5),
                         -2.0 * ((v + 0.5) / height - 0.5),
                         -1.0,
                         1.0);
    }

    let z_clip = (far / (far - near)) * (1.0 - (near / z));

    return vec4<f32>(2.0 * ((u + 0.5) / width - 0.5),
                    -2.0 * ((v + 0.5) / height - 0.5),
                    z_clip,
                    1.0);
}
