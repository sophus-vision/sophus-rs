// Resolves the multisampled depth of one frustum face into its slot of the depth atlas.
//
// The faces are rendered one at a time into a shared depth target, so this runs once per face,
// right after it. The slot is selected by a dynamic offset rather than by writing a uniform per
// face: `Queue::write_buffer` does not interleave with the passes of a command buffer which is
// being encoded, so every face would otherwise resolve into whichever slot was written last.

struct FaceSlot {
    offset_x: u32,
};

@group(0) @binding(0) var<uniform> slot: FaceSlot;
@group(0) @binding(1) var face_depth: texture_multisampled_2d<f32>;
@group(0) @binding(2) var depth_atlas: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn resolve_face_depth(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(face_depth);
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }
    let coords = vec2<u32>(global_id.xy);

    // nearest of all samples, as in the single-plane path (ndc z: 0 is near, 1 is far)
    var ndc_z = textureLoad(face_depth, coords, 0);
    for (var sample = 1; sample < i32(textureNumSamples(face_depth)); sample++) {
        ndc_z = min(ndc_z, textureLoad(face_depth, coords, sample));
    }
    textureStore(depth_atlas, vec2<u32>(coords.x + slot.offset_x, coords.y), ndc_z);
}
