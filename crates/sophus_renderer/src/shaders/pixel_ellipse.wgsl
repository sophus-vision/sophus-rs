@group(0) @binding(0)
var<uniform> camera: CameraProperties;

@group(0) @binding(1)
var<uniform> zoom_2d: Zoom2d;

@group(0) @binding(2)
var<uniform> ortho_camera: PinholeModel;

struct VertexOut {
    @location(0) color: vec4<f32>,
    // offset from the centre, in *image* pixels, and the map taking the ellipse to the unit
    // circle - the fragment shader works out coverage from the two
    @location(1) offset: vec2<f32>,
    @location(2) to_unit_circle: vec4<f32>,
    @location(3) line_width: f32,
    @location(4) feather: f32,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) center: vec2<f32>,
    @location(1) half_extent: vec2<f32>,
    @location(2) to_unit_circle: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) line_width: f32,
    @builtin(vertex_index) idx: u32) -> VertexOut
{
    var out: VertexOut;
    // one view-port pixel, in image pixels - the width of the antialiased edge
    let feather = ortho_camera.viewport_scale;

    // An ellipse is a region of the image, so its extent is zoomed along with its anchor - unlike
    // a point, whose size is a view-port quantity and must not be. The quad is grown by the
    // feather and by the outline, so that the fragments they cover exist at all.
    let anchor = zoom_apply(center, zoom_2d);
    let extent = half_extent * zoom_scaling(zoom_2d) + feather + line_width * feather;

    let mod4 = idx % 6u;
    var corner = vec2<f32>(-1.0, -1.0);
    if mod4 == 1u || mod4 == 3u {
        corner = vec2<f32>(1.0, -1.0);
    } else if mod4 == 2u || mod4 == 4u {
        corner = vec2<f32>(-1.0, 1.0);
    } else if mod4 == 5u {
        corner = vec2<f32>(1.0, 1.0);
    }
    let offset = corner * extent;

    out.position = ortho_pixel_to_clip(
        anchor + offset,
        vec2<f32>(camera.camera_image_width, camera.camera_image_height));
    out.color = color;
    out.offset = offset;
    out.to_unit_circle = to_unit_circle;
    out.line_width = line_width;
    out.feather = feather;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // back to image pixels, undoing the zoom which the offset was grown by
    let local = in.offset / zoom_scaling(zoom_2d);
    let u = vec2<f32>(
        dot(in.to_unit_circle.xy, local),
        dot(in.to_unit_circle.zw, local),
    );

    // How far inside the ellipse this fragment is, in view-port pixels. The margin `1 - |u|` is in
    // units of the ellipse rather than of the image, so it is divided by its own gradient to
    // become a distance - which for the isotropic case is exactly the radius less the offset.
    //
    // That distance is in image pixels, and what it is compared against - the feather and the
    // width of the outline - are in the pixels of the *zoomed* image, which is what the view port
    // shows one for one. Hence the zoom below multiplies: a zoomed-in ellipse covers more of the
    // screen, so a fragment a given number of image pixels inside it is more view-port pixels
    // inside it, not fewer.
    let radius = length(u);
    if (radius < 1e-6) {
        // the centre, which is inside whatever the shape
        return vec4<f32>(in.color.rgb, in.color.a * select(1.0, 0.0, in.line_width > 0.0));
    }
    let direction = u / radius;
    let gradient = vec2<f32>(
        dot(vec2<f32>(in.to_unit_circle.x, in.to_unit_circle.z), direction),
        dot(vec2<f32>(in.to_unit_circle.y, in.to_unit_circle.w), direction),
    );
    let inside = (1.0 - radius) * zoom_scaling(zoom_2d).x / max(length(gradient), 1e-9);

    // filled, or a band of `line_width` view-port pixels just inside the outline - and as a
    // wireframe, always a band
    let line_width = select(
        in.line_width, max(in.line_width, WIREFRAME_WIDTH), ortho_camera.wireframe > 0.5);
    var coverage = clamp(inside / in.feather + 0.5, 0.0, 1.0);
    if (line_width > 0.0) {
        coverage = coverage
            - clamp((inside - line_width * in.feather) / in.feather + 0.5, 0.0, 1.0);
    }
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
