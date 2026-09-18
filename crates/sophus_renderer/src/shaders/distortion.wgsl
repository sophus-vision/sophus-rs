@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(1)
var<uniform> zoom: Zoom2d;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;

@group(1) @binding(0) var input_texture : texture_2d<f32>;
@group(1) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var background_texture : texture_2d<f32>;
@group(1) @binding(3) var depth_texture : texture_multisampled_2d<f32>;
@group(1) @binding(4) var inverse_depth_texture : texture_storage_2d<r32float, write>;
@group(1) @binding(5) var face_atlas : texture_2d<f32>;
@group(1) @binding(6) var face_depth_atlas : texture_2d<f32>;

// The faces of the multi-frustum intermediate, laid out side by side in `face_atlas`. Must agree
// with `Frustum` and `HEMISPHERE` on the cpu side.
const FACE_COUNT: i32 = 5;
const FACE_FORWARD: i32 = 0;
const FACE_RIGHT: i32 = 1;
const FACE_LEFT: i32 = 2;
const FACE_DOWN: i32 = 3;
const FACE_UP: i32 = 4;

// Which face observes `direction`, or -1 when it points backwards.
// The face whose axis `direction` is closest to, or -1 when that is the backward axis. Kept in
// step with `Frustum::covering` on the cpu, which picks the faces to render - a disagreement
// leaves rays reading a face which was never rendered, which
// `every_ray_is_read_back_from_a_frustum_which_was_rendered` checks for, pixel by pixel.
fn face_covering(direction: vec3<f32>) -> i32 {
    let ax = abs(direction.x);
    let ay = abs(direction.y);
    if (-direction.z > ax && -direction.z > ay) {
        return -1;
    }
    if (direction.z >= ax && direction.z >= ay) {
        return FACE_FORWARD;
    }
    if (ax >= ay) {
        return select(FACE_LEFT, FACE_RIGHT, direction.x > 0.0);
    }
    return select(FACE_UP, FACE_DOWN, direction.y > 0.0);
}

// `direction`, expressed in the frame of `face` - whose own optical axis is +z.
fn direction_in_face(face: i32, d: vec3<f32>) -> vec3<f32> {
    switch face {
        case 1: { return vec3<f32>(-d.z, d.y, d.x); }
        case 2: { return vec3<f32>(d.z, d.y, -d.x); }
        case 3: { return vec3<f32>(d.x, -d.z, d.y); }
        case 4: { return vec3<f32>(d.x, d.z, -d.y); }
        default: { return d; }
    }
}

// A colour per frustum, so that the layout of the faces can be seen in the distorted image.
// The colour of the one plane, told apart from the five faces by being neutral rather than by
// being a paler shade of them - a difference in brightness reads as "the same thing, dimmer".
const SINGLE_PLANE_TINT = vec3<f32>(0.55, 0.55, 0.55);
const TINT_STRENGTH = 0.55;

fn frustum_tint(face: i32) -> vec3<f32> {
    switch face {
        case 0: { return vec3<f32>(0.20, 0.55, 1.00); }
        case 1: { return vec3<f32>(1.00, 0.45, 0.20); }
        case 2: { return vec3<f32>(0.25, 0.85, 0.40); }
        case 3: { return vec3<f32>(0.95, 0.85, 0.20); }
        case 4: { return vec3<f32>(0.80, 0.40, 0.90); }
        default: { return vec3<f32>(0.55, 0.55, 0.55); }
    }
}

// Where `direction` lands in the atlas: the column of its face plus the position within it.
// Returns a negative x when the direction points backwards, out of every face.
fn atlas_coords_of(direction: vec3<f32>) -> vec2<f32> {
    let face = face_covering(direction);
    // Only the faces which were rendered this frame: the atlas keeps one slot per face of the
    // hemisphere, and the slots of the faces left out hold stale pixels of an earlier frame - or,
    // the first time round, nothing at all. Rust picks the faces so that every visible ray has
    // one; a ray without is a gap in that cover, and shows the background rather than garbage.
    if (face < 0 || (pinhole.rendered_faces & (1u << u32(face))) == 0u) {
        return vec2<f32>(-1.0, -1.0);
    }
    // every face is a 90 degree frustum of `pinhole.width` pixels a side
    let face_size = pinhole.width;
    let in_face = direction_in_face(face, direction);
    let focal = 0.5 * face_size;
    let center = 0.5 * (face_size - 1.0);
    let uv = vec2<f32>(
        focal * in_face.x / in_face.z + center,
        focal * in_face.y / in_face.z + center,
    );
    // clamped inside this face, so that it cannot bleed into its neighbour
    let clamped = clamp(uv, vec2<f32>(0.0, 0.0), vec2<f32>(face_size - 1.0, face_size - 1.0));
    return vec2<f32>(f32(face) * face_size + clamped.x, clamped.y);
}

// Colour observed along `direction`, out of the face which covers it.
fn sample_faces(atlas_uv: vec2<f32>) -> vec4<f32> {
    if (atlas_uv.x < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    // bilinear, but not across the seam between two faces
    let face_size = pinhole.width;
    let face_start = floor(atlas_uv.x / face_size) * face_size;
    let p0 = floor(atlas_uv);
    let p1 = vec2<f32>(
        min(p0.x + 1.0, face_start + face_size - 1.0),
        min(p0.y + 1.0, face_size - 1.0),
    );
    let t = atlas_uv - p0;

    let c00 = textureLoad(face_atlas, vec2<u32>(vec2<f32>(p0.x, p0.y)), 0);
    let c10 = textureLoad(face_atlas, vec2<u32>(vec2<f32>(p1.x, p0.y)), 0);
    let c01 = textureLoad(face_atlas, vec2<u32>(vec2<f32>(p0.x, p1.y)), 0);
    let c11 = textureLoad(face_atlas, vec2<u32>(vec2<f32>(p1.x, p1.y)), 0);
    return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

// Composites the scene, sampled out of the faces, over `background_color`.
fn distort_pixel_from_faces(
    view_port_coords_distorted: vec2<u32>,
    image_size: vec2<f32>,
    view_port_size: vec2<u32>,
    background_color: vec4<f32>
) {
    let uv_distorted = (vec2<f32>(view_port_coords_distorted) + 0.5) * vec2<f32>(image_size)
        / vec2<f32>(view_port_size) - 0.5;
    let direction = normalize(distorted_to_ray(uv_distorted, camera));
    let atlas_uv = atlas_coords_of(direction);
    let foreground_color = sample_faces(atlas_uv);

    // Depth of the same face at the same place - without interpolation, as in the single plane
    // path, since an interpolated depth between two surfaces belongs to neither.
    var inverse_distance = 0.0;
    let face = face_covering(direction);
    if (atlas_uv.x >= 0.0) {
        let ndc_in_face = textureLoad(face_depth_atlas, vec2<u32>(atlas_uv), 0).r;
        inverse_distance = inverse_depth_along_ray(
            ndc_in_face, direction_in_face(face, direction).z);
    }

    var rgb = foreground_color.rgb + background_color.rgb * (1.0 - foreground_color.a);
    if (pinhole.debug_frustum_planes > 0.5) {
        // The face this pixel was actually read from, which is `face` only for the faces which
        // were rendered. Anything else is a hole in the cover - a ray Rust did not expect to be
        // looking at - and shows up as the neutral tint rather than as a plausible colour.
        let sampled = select(-1, face, atlas_uv.x >= 0.0);
        rgb = mix(rgb, frustum_tint(sampled), TINT_STRENGTH);
    }

    // Traced primitives are not part of any intermediate: they are intersected with this
    // pixel's own ray, and composited by distance against whatever was rasterized.
    let traced_pixel = composite_traced(
        rgb,
        inverse_distance,
        direction,
        neighbouring_rays(uv_distorted, image_size, view_port_size));

    textureStore(inverse_depth_texture, view_port_coords_distorted,
                 vec4<f32>(traced_pixel.inverse_distance, 0.0, 0.0, 0.0));
    textureStore(output_texture, view_port_coords_distorted,
                 vec4<f32>(traced_pixel.rgb, 1.0));
}

 fn distort_pixel(
    view_port_coords_distorted: vec2<u32>,
    image_size: vec2<f32>,
    view_port_size: vec2<u32>,
    background_color: vec4<f32>
) {
    // Note: `camera` and `pinhole` carry the 2d zoom (folded into the intrinsics), hence
    // `uv_distorted` is a zoomed - i.e. view-port ("screen") - coordinate, and so is the
    // undistorted result. The scene texture was rendered through the very same zoomed models.
    // Pixel *centers*: the image coordinate `k` is the center of image pixel `k`, which is the
    // convention of `pixel_and_z_to_clip` / `ortho_pixel_to_clip` and of `ViewportScale` on the
    // CPU side. Treating `k` as a pixel corner here instead differs by half a pixel, which
    // cancels at scale 1 but drifts apart with the 2d zoom - the background image would then
    // slide away from the 2d renderables and the 3d augmentations as the view is zoomed.
    let uv_distorted = (vec2<f32>(view_port_coords_distorted) + 0.5) * vec2<f32>(image_size)
        / vec2<f32>(view_port_size) - 0.5;
    // the ray through this pixel, in the camera frame: the undistorted pixel is read off it,
    // and so is the distance to whatever the plane holds there
    let ray = distorted_to_ray(uv_distorted, camera);
    let uv_undistorted = z1_plane_to_undistorted(ray.xy / ray.z, pinhole).xy;
    let view_port_coords_undistorted = (uv_undistorted + 0.5) * vec2<f32>(view_port_size)
        / vec2<f32>(image_size) - 0.5;

    // Outside the rendered scene texture there is nothing of the plane to sample - but a traced
    // primitive is not part of the plane, so this goes on to the compositing below rather than
    // storing the background and returning.
    let outside = view_port_coords_undistorted.x < 0.0
        || view_port_coords_undistorted.y < 0.0
        || view_port_coords_undistorted.x > f32(view_port_size.x) - 1.0
        || view_port_coords_undistorted.y > f32(view_port_size.y) - 1.0;

    // bi-linear interpolation. Clamped, since a pixel outside the plane samples nothing - what
    // it reads is thrown away below.
    let x0 = floor(clamp(view_port_coords_undistorted.x, 0.0, f32(view_port_size.x) - 1.0));
    let x1 = ceil(clamp(view_port_coords_undistorted.x, 0.0, f32(view_port_size.x) - 1.0));
    let y0 = floor(clamp(view_port_coords_undistorted.y, 0.0, f32(view_port_size.y) - 1.0));
    let y1 = ceil(clamp(view_port_coords_undistorted.y, 0.0, f32(view_port_size.y) - 1.0));

    let tx = clamp(view_port_coords_undistorted.x, 0.0, f32(view_port_size.x) - 1.0) - x0;
    let ty = clamp(view_port_coords_undistorted.y, 0.0, f32(view_port_size.y) - 1.0) - y0;

    let c00 = textureLoad(input_texture, vec2<u32>(u32(x0), u32(y0)), 0);
    let c10 = textureLoad(input_texture, vec2<u32>(u32(x1), u32(y0)), 0);
    let c01 = textureLoad(input_texture, vec2<u32>(u32(x0), u32(y1)), 0);
    let c11 = textureLoad(input_texture, vec2<u32>(u32(x1), u32(y1)), 0);

    var foreground_color = mix(
        mix(c00, c10, tx),
        mix(c01, c11, tx),
        ty
    );
    if (outside) {
        foreground_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // depth lookup without interpolation, taking the nearest of all samples
    // (ndc z: 0 is the near plane, 1 the far plane)
    let depth_coords = vec2<u32>(view_port_coords_undistorted);
    var ndc_z = textureLoad(depth_texture, depth_coords, 0).r;
    for (var sample = 1; sample < i32(textureNumSamples(depth_texture)); sample++) {
        ndc_z = min(ndc_z, textureLoad(depth_texture, depth_coords, sample).r);
    }
    // this intermediate is a plane whose optical axis is the camera's own
    var inverse_distance = inverse_depth_along_ray(ndc_z, normalize(ray).z);
    if (outside) {
        inverse_distance = 0.0;
    }

    // The scene texture holds *premultiplied* colors (see the scene fragment shaders), so the
    // foreground is added rather than mixed in - `mix` would darken every antialiased edge,
    // because multisample resolve turns partial coverage into a premultiplied color.
    var rgb = foreground_color.rgb + background_color.rgb * (1.0 - foreground_color.a);
    if (pinhole.debug_frustum_planes > 0.5) {
        // One flat tint: the scene came off a single plane, so there is one intermediate and no
        // seams to point at. The five colours below mean five frusta - nothing else.
        rgb = mix(rgb, SINGLE_PLANE_TINT, TINT_STRENGTH);
    }

    let direction = normalize(ray);
    let traced_pixel = composite_traced(
        rgb,
        inverse_distance,
        direction,
        neighbouring_rays(uv_distorted, image_size, view_port_size));

    textureStore(inverse_depth_texture, view_port_coords_distorted,
                 vec4<f32>(traced_pixel.inverse_distance, 0.0, 0.0, 0.0));
    // Keep the output fully opaque
    textureStore(output_texture, view_port_coords_distorted,
                 vec4<f32>(traced_pixel.rgb, 1.0));
}

@compute @workgroup_size(16, 16)
fn distort(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let view_port_size = textureDimensions(input_texture);
    let coords = vec2<u32>(global_id.xy);

    if (coords.x >= view_port_size.x || coords.y >=  view_port_size.y) {
        return;
    }

    let background_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let image_size = vec2<f32>(camera.camera_image_width, camera.camera_image_height);

    distort_pixel(coords, image_size, view_port_size, background_color);
}

@compute @workgroup_size(16, 16)
fn distort_with_background(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let view_port_size = textureDimensions(input_texture);
    let coords = vec2<u32>(global_id.xy);

    if (coords.x >= view_port_size.x || coords.y >=  view_port_size.y) {
        return;
    }

    let image_size = vec2<f32>(camera.camera_image_width, camera.camera_image_height);
    // view-port ("screen") coordinate of this output pixel, in pixel-center convention ...
    let uv_screen = (vec2<f32>(global_id.xy) + 0.5) * vec2<f32>(image_size)
        / vec2<f32>(view_port_size) - 0.5;
    // ... and the pixel of the (unzoomed) background image which is displayed there. Rounding to
    // the nearest index, since `k` denotes the center of image pixel `k`.
    let image_index = zoom_apply_inv(uv_screen, zoom) + 0.5;

    var background_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (image_index.x >= 0.0 && image_index.y >= 0.0
        && image_index.x < image_size.x && image_index.y < image_size.y) {
        background_color = textureLoad(background_texture, vec2<u32>(image_index), 0);
    }

    distort_pixel(coords, image_size, view_port_size, background_color);
}

@compute @workgroup_size(16, 16)
fn distort_faces(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let view_port_size = textureDimensions(output_texture);
    let coords = vec2<u32>(global_id.xy);
    if (coords.x >= view_port_size.x || coords.y >= view_port_size.y) {
        return;
    }
    let image_size = vec2<f32>(camera.camera_image_width, camera.camera_image_height);
    distort_pixel_from_faces(
        coords, image_size, view_port_size, vec4<f32>(1.0, 1.0, 1.0, 1.0));
}

@compute @workgroup_size(16, 16)
fn distort_faces_with_background(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let view_port_size = textureDimensions(output_texture);
    let coords = vec2<u32>(global_id.xy);
    if (coords.x >= view_port_size.x || coords.y >= view_port_size.y) {
        return;
    }

    let image_size = vec2<f32>(camera.camera_image_width, camera.camera_image_height);
    let uv_screen = (vec2<f32>(global_id.xy) + 0.5) * vec2<f32>(image_size)
        / vec2<f32>(view_port_size) - 0.5;
    let image_index = zoom_apply_inv(uv_screen, zoom) + 0.5;

    var background_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (image_index.x >= 0.0 && image_index.y >= 0.0
        && image_index.x < image_size.x && image_index.y < image_size.y) {
        background_color = textureLoad(background_texture, vec2<u32>(image_index), 0);
    }

    distort_pixel_from_faces(coords, image_size, view_port_size, background_color);
}
