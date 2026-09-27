@group(0) @binding(0) var src_bgra: texture_2d<f32>;              
@group(0) @binding(1) var src_bgra_sampler: sampler;
@group(0) @binding(2) var y_plane: texture_storage_2d<r8unorm, write>;    
@group(0) @binding(3) var cbcr_plane: texture_storage_2d<rg8unorm, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let transform = mat3x3<f32>(
        vec3(0.2126, -0.1146, 0.5),
        vec3(0.7152, -0.3854, -0.4542),
        vec3(0.0722, 0.5, -0.0458)
    );

    let base = vec2<u32>(gid.xy) * 2;

    let cbcr_dims = textureDimensions(cbcr_plane);
    let dims = textureDimensions(src_bgra);
    if (base.x >= dims.x || base.y >= dims.y) { return; }

    var cb_sum = 0.0;
    var cr_sum = 0.0;
    var count = 0.0;

    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let coord = base + vec2<u32>(u32(dx), u32(dy));
            if (coord.x >= dims.x || coord.y >= dims.y) { continue; }

            let px = textureLoad(src_bgra, coord, 0);
            let ycbcr = transform * px.rgb + vec3(0., 0.5, 0.5);

            textureStore(y_plane, coord, vec4(ycbcr.r, 0., 0., 0.));

            cb_sum += ycbcr.g;
            cr_sum += ycbcr.b;
            count += 1.0;
        }
    }

    let avg = vec4(cb_sum / count, cr_sum / count, 0., 0.);
    textureStore(cbcr_plane, vec2<u32>(gid.xy), avg);
}
