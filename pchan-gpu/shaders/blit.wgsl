struct Uniforms {
    kind: u32,
    gpustat: u32,
    src: vec2<u32>,
    dest: vec2<u32>,
    size: vec2<u32>
}

struct Push {
    cursor: u32,
    index: u32,
}

var<immediate> push: Push;
@group(0) @binding(0) var<storage> uniforms_buf: array<Uniforms>;
@group(0) @binding(1) var<storage, read> src_buf: array<u32>;
@group(0) @binding(2) var dest_tex: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(3) var vram_tex: texture_storage_2d<rgba8unorm,read_write>;

const CPU_TO_VRAM: u32 = 0;
const VRAM_TO_VRAM: u32 = 1;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cursor = push.cursor;
    let index = push.index;
    let uniforms = uniforms_buf[index];

    var size = copy_clip_size(uniforms.size);
    var dest = clip_pos(gid.xy+uniforms.dest);

    if (any(gid.xy >= size)) { return; }

    var pbw = get_flag(uniforms.gpustat, 11);
    var pbc = get_flag(uniforms.gpustat, 12);
    let current = textureLoad(vram_tex, vec2<i32>(dest));
    if pbc && current.a > 0.5 {
        return;
    }

    switch uniforms.kind {
        case CPU_TO_VRAM {
            let i0 = gid.y * size.x + gid.x + cursor / 2u;
            var value = rgb5_split_color(get_packed_src(i0));

            if pbw {
                value.a = 1.0;
            }

            textureStore(dest_tex, dest, value);
            textureStore(vram_tex, dest, value);
        }
        case VRAM_TO_VRAM, default: {
            let sp = vec2(gid.x, gid.y);
            var value = get_vram(sp);

            if pbw {
                value.a = 1.0;
            }

            textureStore(dest_tex, dest, value);
            textureStore(vram_tex, dest, value);
            // textureStore(dest_tex, dest_left, vec4(0x0000, 0, 0, 0));
            // textureStore(dest_tex, dest_right, vec4(0x0000, 0, 0, 0));
            // textureStore(vram_tex, vramcoord_to_texcoord(dest), vec4(0x00000000, 0, 0, 0));
        }
    }
}

fn vramcoord_to_texcoord(coord: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(vec2(coord.x/2, coord.y));
}

fn read_16bit(coord: vec2<u32>) -> vec4<f32> {
    let texcoord = vramcoord_to_texcoord(coord);
    let index = texcoord.y*1024*2+texcoord.x*2;
    // wgpu tex coords are +Y = down
    let texel = unpack4x8unorm(src_buf[index]);
    return texel;
}

fn read_raw16(coord: vec2<u32>) -> u32 {
    let t = read_16bit(coord);
    let r = u32(round(t.r * 31.0));
    let g = u32(round(t.g * 31.0));
    let b = u32(round(t.b * 31.0));
    let a = u32(round(t.a));
    return r | (g << 5u) | (b << 10u) | (a << 15u);
}

fn clip_pos(x: vec2<u32>) -> vec2<u32> {
    return x & vec2(0x3FFu, 0x1FFu);
}

fn get_vram(p: vec2<u32>) -> vec4<f32> {
    let q = clip_pos(p);
    return unpack4x8unorm(src_buf[q.y * 1024u + q.x]);
}

fn copy_clip_size(x: vec2<u32>) -> vec2<u32> {
    var size = x;
    if size.x == 0 {
        size.x = 1024;
    }
    if size.y == 0 {
        size.y = 512;
    }
    size = ((size - 1u) & vec2(0x3FFu, 0x1FFu)) + 1u;
    return size;
}

fn get_packed_src(idx: u32) -> u32 {
    return (src_buf[idx / 2] >> ((idx & 1) * 16)) & 0xFFFF;
}


fn pos_to_idx(pos: vec2<u32>, size: vec2<u32>) -> u32 {
    return pos.x + pos.y * size.x;
}

fn get_flag(flags: u32, idx: u32) -> bool {
    return (flags & (u32(1) << idx)) != 0;
}

fn get_mask_bit(color: u32) -> bool {
    return get_flag(color, 15);
}

fn rgb5_set_stp(value: u32) -> u32 {
    let c = value | (1 << 15u);
    return c;
}

fn rgb5_split_color(value: u32) -> vec4<f32> {
    let r = f32(value & 0x1Fu) / 31.0;
    let g = f32((value >> 5u) & 0x1Fu) / 31.0;
    let b = f32((value >> 10u) & 0x1Fu) / 31.0;
    let stp = f32(get_flag(value, 15));
    return vec4(r, g, b, stp);
}
