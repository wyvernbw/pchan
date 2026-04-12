struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
    @location(1) uv: vec2<f32>
}

@group(0) @binding(0)
var render_t: texture_2d<u32>;
@group(0) @binding(1)
var render_s: sampler;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos: vec2<f32>;
    var vertices = array<vec2<f32>, 6>(
        vec2(-1, -1),
        vec2(-1, 1),
        vec2(1, 1),
        vec2(-1, -1),
        vec2(1, 1),
        vec2(1, -1)
    );
    pos = vertices[in_vertex_index];
    out.clip_position = vec4<f32>(pos.x, pos.y, 0.0, 1.0);
    out.vert_pos = out.clip_position.xyz;
    out.uv = (out.clip_position.xy + vec2(1)) * vec2(0.5);
    return out;
}

fn read_16bit(coord: vec2<f32>) -> u32 {
    let texcoord = vec2<u32>(u32(coord.x), u32(coord.y));
    return textureLoad(render_t, texcoord, 0).r;
}

fn rgb5_split_color(value: u32) -> vec3<f32> {
    let r = f32(value & 0x1Fu) / 31.0;
    let g = f32((value >> 5u) & 0x1Fu) / 31.0;
    let b = f32((value >> 10u) & 0x1Fu) / 31.0;
    return vec3(r, g, b);
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.uv * vec2(1024, 512);
    uv.y = 512 - uv.y;
    var col = read_16bit(uv);
    var out = rgb5_split_color(col);
    out.r = srgb_to_linear(out.r);
    out.g = srgb_to_linear(out.g);
    out.b = srgb_to_linear(out.b);

    return vec4<f32>(out, 1.0);
}

