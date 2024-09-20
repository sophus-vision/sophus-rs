@group(0) @binding(0)
var<uniform> camera: CameraProperties;
@group(0) @binding(1)
var<uniform> zoom: Zoom2d;
@group(0) @binding(2)
var<uniform> pinhole: PinholeModel;

@group(1) @binding(0) var input_texture : texture_2d<f32>;
@group(1) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var background_texture : texture_2d<f32>;
@group(1) @binding(3) var depth_texture : texture_2d<f32>;
@group(1) @binding(4) var distorted_depth_texture : texture_storage_2d<r32float, write>;

 fn distort_pixel(
    view_port_coords_distorted: vec2<u32>,
    image_size: vec2<f32>,
    view_port_size: vec2<u32>,
    background_color: vec4<f32>
) {
        let uv_distorted =  vec2<f32>(view_port_coords_distorted) *  vec2<f32>(image_size) / vec2<f32>(view_port_size);
        let uv_undistorted = undistort(vec2<f32>(uv_distorted), pinhole, camera);
        let view_port_coords_undistorted = uv_undistorted * vec2<f32>(view_port_size) / vec2<f32>(image_size);

        // bi-linear interpolation
        let x0 = floor(view_port_coords_undistorted.x);
        let x1 = ceil(view_port_coords_undistorted.x);
        let y0 = floor(view_port_coords_undistorted.y);
        let y1 = ceil(view_port_coords_undistorted.y);

        let tx = view_port_coords_undistorted.x - x0;
        let ty = view_port_coords_undistorted.y - y0;

        let c00 = textureLoad(input_texture, vec2<u32>(u32(x0), u32(y0)), 0);
        let c10 = textureLoad(input_texture, vec2<u32>(u32(x1), u32(y0)), 0);
        let c01 = textureLoad(input_texture, vec2<u32>(u32(x0), u32(y1)), 0);
        let c11 = textureLoad(input_texture, vec2<u32>(u32(x1), u32(y1)), 0);

        let foreground_color = mix(
            mix(c00, c10, tx),
            mix(c01, c11, tx),
            ty
        );

        // depth lookup without interpolation
        let depth = textureLoad(depth_texture,  vec2<u32>(view_port_coords_undistorted), 0);

        // Use the alpha channel of the foreground for blending
        let alpha = foreground_color.a;

        let mixed = vec4<f32>(
            mix(background_color.rgb, foreground_color.rgb, alpha),
            1.0 // Keep the output fully opaque
        );

        textureStore(distorted_depth_texture, view_port_coords_distorted, depth);
        textureStore(output_texture, view_port_coords_distorted, mixed);
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
    let image_coords =  vec2<f32>(global_id.xy) *  vec2<f32>(image_size) / vec2<f32>(view_port_size);
    let image_coords_floor = vec2<u32>(image_coords);

    let background_color = textureLoad(background_texture, image_coords_floor, 0);

    distort_pixel(coords, image_size, view_port_size, background_color);
}
