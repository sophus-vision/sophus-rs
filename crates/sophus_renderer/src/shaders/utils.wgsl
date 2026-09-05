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
    // The direction the light shines, in the entity's own frame - which is the frame its normals
    // are given in, so that shading needs no transform. The cpu rotates it there because the pose
    // above is the one the *intermediate* is rendered from: for a frustum face that is not the
    // camera, and a light held in the rendering frame would swing from face to face and seam.
    light_in_entity: vec4<f32>,
    // 1 to draw this entity as edges, whatever the rest of the scene does
    wireframe: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
};

struct PinholeModel {
     width: f32,
     height: f32,
     fx: f32,
     fy: f32,
     px: f32,
     py: f32,
     viewport_scale: f32,
     /// 1 to tint the image by which frustum each pixel came from, 0 otherwise
     debug_frustum_planes: f32,
     /// bit `i` is set when face `i` was rendered into the atlas this frame
     rendered_faces: u32,
     /// 1 to draw everything as edges rather than as surfaces, 0 otherwise
     wireframe: f32,
     padding0: u32,
     padding1: u32,
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
    camera: CameraProperties) -> Projection
{
   var out: Projection;
   out.point_in_z1 = scene_point_to_z1_plane_and_depth(point, view).xy;
   out.uv_undistorted = z1_plane_to_undistorted(out.point_in_z1, pinhole);
   out.uv = z1_plane_to_distorted(out.point_in_z1, camera);
   out.z = scene_point_to_z1_plane_and_depth(point, view).z;
   return out;
}

// Maps a distorted pixel to the *ray* it observes, as a direction in the camera frame.
//
// This is where `distorted_to_z1` gets its `k` from: the ray is `(u, v, k)`. Dividing by `k` to
// land on the z = 1 plane loses exactly the rays this returns fine - `k` goes to zero at 90
// degrees off axis, so a field of view approaching 180 degrees has no z = 1 representation.
fn distorted_to_ray(uv_distorted: vec2<f32>, camera: CameraProperties) -> vec3<f32> {
    let u = (uv_distorted.x-camera.px)/camera.fx;
    let v = (uv_distorted.y-camera.py)/camera.fy;

    let r2 = u*u + v*v;
    let gamma = 1.0 - camera.alpha;

    let nominator = 1.0 - camera.alpha * camera.alpha * camera.beta * r2;
    let denominator = camera.alpha * sqrt(1.0 - (camera.alpha - gamma) * camera.beta * r2) + gamma;

    let k = nominator / denominator;

    return vec3<f32>(u, v, k);
}

fn distorted_to_z1(uv_distorted: vec2<f32>, camera: CameraProperties) -> vec2<f32> {
    let ray = distorted_to_ray(uv_distorted, camera);
    return ray.xy / ray.z;
}

// Maps a distorted pixel to the pixel of the undistorted (pinhole) intermediate which observes
// the same ray. Only defined while that ray is in front of the pinhole intermediate - a single
// plane cannot hold a ray at 90 degrees off axis, see `distorted_to_ray`.
fn undistort(uv_distorted: vec2<f32>, pinhole: PinholeModel, camera: CameraProperties) -> vec2<f32> {
    let ray = distorted_to_ray(uv_distorted, camera);
    return z1_plane_to_undistorted(ray.xy / ray.z, pinhole).xy;
}

// apply the 2d zoom: image pixel coordinates -> view-port ("screen") coordinates
//
// Both spaces are in units of image pixels; the view port always shows the rect
// [0, image_width] x [0, image_height] of the zoomed space.
// the scaling of the 2d zoom, as a vector
fn zoom_scaling(zoom: Zoom2d) -> vec2<f32> {
    return vec2<f32>(zoom.scaling_x, zoom.scaling_y);
}

fn zoom_apply(uv: vec2<f32>, zoom: Zoom2d) -> vec2<f32> {
    return vec2<f32>(uv.x * zoom.scaling_x + zoom.translation_x,
                     uv.y * zoom.scaling_y + zoom.translation_y);
}

// inverse of `zoom_apply`: view-port ("screen") coordinates -> image pixel coordinates
fn zoom_apply_inv(uv: vec2<f32>, zoom: Zoom2d) -> vec2<f32> {
    return vec2<f32>((uv.x - zoom.translation_x) / zoom.scaling_x,
                     (uv.y - zoom.translation_y) / zoom.scaling_y);
}

// convert from pixel to clip space
// Places an image pixel of the *final, distorted* image - where 2d renderables are anchored -
// in clip space. Note it takes the image size rather than a `PinholeModel`: the pinhole model
// describes the intermediate the scene is rasterized into, which is a 90 degree face when the
// view is rendered through the frusta, and nothing to do with the size of the image.
// Inverse distance of a pixel, from the depth an intermediate wrote for it.
//
// An intermediate - the single plane, or one of the frustum faces - measures depth along its own
// optical axis and encodes it the way the rasterizer does, hyperbolically between the near and
// far planes. `axis_component` is the component of the (unit) ray through the pixel along that
// axis, which turns that depth into a distance along the ray, and the inverse of it is what the
// inverse distance image holds: zero for a pixel holding nothing, which is the same thing as a
// surface infinitely far away.
//
// Note that nothing is clamped to the clipping planes here. What is visible was decided when the
// intermediate was rasterized, and a surface within the far plane along the optical axis can be
// further than the far plane away from the camera.
fn inverse_depth_along_ray(ndc_z: f32, axis_component: f32) -> f32 {
    if (ndc_z >= 1.0 || axis_component <= 0.0) {
        return 0.0;
    }
    let z = camera.near / (1.0 - ndc_z * (camera.far - camera.near) / camera.far);
    return axis_component / z;
}

// Ambient floor, so a surface turned away from the light goes dim rather than black.
//
// Fairly high, because the light is fixed to the camera: a ground plane is then lit edge on
// whatever the view, and a floor which is always at the floor of the shading reads as black.
const SHADING_AMBIENT: f32 = 0.62;

// Lambert against the light, plus that floor. Both vectors must be in the same frame; the pose
// uniform carries the light in the entity frame, which is where normals live.
//
// Two-sided, deliberately: normals come from the winding of the triangle, and backface culling is
// off by default, so an open mesh - a single triangle, say - would otherwise show one of its
// sides at the ambient floor and nothing more.
fn shade(normal: vec3<f32>, light_direction: vec3<f32>, rgb: vec3<f32>) -> vec3<f32> {
    let lambert = abs(dot(normalize(normal), normalize(light_direction)));
    return rgb * (SHADING_AMBIENT + (1.0 - SHADING_AMBIENT) * lambert);
}

// Whether the scene already put something nearer than this at the pixel being drawn.
//
// The distortion pass leaves the inverse distance along the ray of every pixel behind it, so anything
// drawn over the finished image can be occluded by the scene without being part of it. Zero means
// the scene left nothing there, and nothing never occludes.
//
// The margin lets a thing sitting exactly on a surface - a point marking a landmark on it, say -
// stay visible, where an exact test would leave it flickering against its own footing.
fn occluded(inverse_depth_texture: texture_2d<f32>, at: vec2<f32>, inverse_distance: f32) -> bool {
    let scene = textureLoad(inverse_depth_texture, vec2<i32>(at), 0).r;
    return scene > inverse_distance * 1.02;
}

// Width of a wireframe line, in view-port pixels. Two rather than one: a hairline reads as a
// smudge against a busy scene, and a traced primitive drawn as edges has only its silhouette to
// show, so what little it draws has to carry.
const WIREFRAME_WIDTH: f32 = 2.0;

// Whether a thing is drawn as edges: either the view says so, or the thing itself does. The view's
// switch is for looking at a whole scene; the per-entity one is for opening up a single thing while
// the rest of it stays solid.
// Both are passed in rather than read from a uniform, because the 2d shaders bind the same
// uniform under another name.
fn draws_as_wireframe(view_wireframe: f32, entity_wireframe: f32) -> bool {
    return view_wireframe > 0.5 || entity_wireframe > 0.5;
}

// The barycentric coordinate of a corner of a triangle, from the index of its vertex.
//
// The scene meshes are triangle lists with a vertex each, so the corner is the vertex index modulo
// three - there is nothing to store and nothing to change about the vertex data.
fn barycentric_of_corner(vertex_index: u32) -> vec3<f32> {
    let corner = vertex_index % 3u;
    return vec3<f32>(
        select(0.0, 1.0, corner == 0u),
        select(0.0, 1.0, corner == 1u),
        select(0.0, 1.0, corner == 2u),
    );
}

// How much of this fragment a wireframe edge covers: one on the edges of the triangle, nought
// inside it.
//
// The distance to the nearest edge is the smallest barycentric coordinate, in units of how fast
// that coordinate changes across the screen - which `fwidth` gives, this being a fragment shader
// and not the compute pass the tracer runs in.
fn wireframe_coverage(barycentric: vec3<f32>) -> f32 {
    let in_pixels = barycentric / fwidth(barycentric);
    let to_edge = min(in_pixels.x, min(in_pixels.y, in_pixels.z));
    return clamp(WIREFRAME_WIDTH - to_edge + 0.5, 0.0, 1.0);
}

fn ortho_pixel_to_clip(uv: vec2<f32>, image_size: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(2.0 * (uv.x + 0.5) / image_size.x - 1.0,
                     2.0 - 2.0 * (uv.y + 0.5) / image_size.y - 1.0,
                     0.0,
                     1.0);
}

// map point from pixel coordinates (Computer Vision convention) to clip space coordinates (WebGPU convention)
//
// Note: the 2d zoom is not applied here. It is folded into the intrinsics (see
// `VertexShaderUniformBuffers::update`), so that the zoomed scene is rasterized at full view-port
// resolution and the subsequent distortion pass stays consistent with it.
fn pixel_and_z_to_clip(
    uv_z: vec2<f32>,
    z: f32,
    camera: CameraProperties,
    pinhole: PinholeModel) -> vec4<f32>
{
    // the raster target of the scene pass is the *undistorted* image, whose size is the one of
    // the pinhole model - not necessarily the size of the distorted camera image
    var width = pinhole.width;
    var height = pinhole.height;
    var near = camera.near;
    var far = camera.far;
    var u = uv_z.x;
    var v = uv_z.y;

    let ndc_x = 2.0 * ((u + 0.5) / width - 0.5);
    let ndc_y = -2.0 * ((v + 0.5) / height - 0.5);

    // Emit *homogeneous* clip coordinates - `w` is the camera-space depth, not 1.
    //
    // The projection is carried out in the vertex shader (the camera model is not a 4x4 matrix),
    // so it is tempting to emit the already divided position with `w = 1`. But then the
    // rasterizer interpolates varyings and depth linearly in screen space rather than
    // hyperbolically: texture coordinates of a slanted surface come out affine - visibly warped,
    // with a seam along the diagonal of a quad - and interior depth values are wrong.
    //
    // Scaling by `z` is exact: the undistorted pixel coordinate is `u = fx * x/z + px`, hence
    // `u * z = fx * x + px * z` is affine in the vertex position. The hardware divide recovers
    // exactly the same ndc position, but now interpolates correctly.
    // Note: geometry closer than the near plane is *clipped*, not pinned to it. With homogeneous
    // coordinates the hardware does that correctly, including interpolating a triangle which
    // straddles the plane. Pinning it instead placed it at a projected position which runs away
    // as z goes to zero - which a multi-frustum intermediate hits routinely, since geometry well
    // inside one face lies near the plane of its neighbour.
    let z_clip = (far / (far - near)) * (z - near);

    return vec4<f32>(ndc_x * z, ndc_y * z, z_clip, z);
}
