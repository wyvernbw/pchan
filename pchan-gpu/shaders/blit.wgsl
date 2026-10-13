struct Uniforms {
    kind: u32,
    gpustat: u32,
    src: vec2<u32>,
    dest: vec2<u32>,
    size: vec2<u32>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> src_buf: array<u32>;
@group(0) @binding(2) var dest_tex: texture_storage_2d<r16uint,write>;
@group(0) @binding(3) var vram_tex: texture_storage_2d<r16uint,read_write>;

const CPU_TO_VRAM: u32 = 0;
const VRAM_TO_VRAM: u32 = 1;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var size = copy_clip_size(uniforms.size);
    var dest = clip_pos(gid.xy+uniforms.dest);

    if (any(gid.xy >= size)) { return; }

    var pbw = get_flag(uniforms.gpustat, 11);
    var pbc = get_flag(uniforms.gpustat, 12);
    let current = get_vram(dest);
    if pbc && get_mask_bit(current) {
        return;
    }

    switch uniforms.kind {
        case CPU_TO_VRAM {
            let i0 = gid.y * size.x + gid.x;
            var value = get_src(i0);

            if pbw {
                value = rgb5_set_stp(value);
            }

            textureStore(dest_tex, dest, vec4(value, 0, 0, 0));
            textureStore(vram_tex, dest, vec4(value, 0, 0, 0));
        }
        case VRAM_TO_VRAM, default: {
            let sp = vec2(gid.x, gid.y) + uniforms.src;
            var value = get_vram(sp);

            if pbw {
                value = rgb5_set_stp(value);
            }

            textureStore(dest_tex, dest, vec4(value, 0, 0, 0));
            textureStore(vram_tex, dest, vec4(value, 0, 0, 0));
            // textureStore(dest_tex, dest_left, vec4(0x0000, 0, 0, 0));
            // textureStore(dest_tex, dest_right, vec4(0x0000, 0, 0, 0));
            // textureStore(vram_tex, vramcoord_to_texcoord(dest), vec4(0x00000000, 0, 0, 0));
        }
    }
}

fn vramcoord_to_texcoord(coord: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(vec2(coord.x/2, coord.y));
}

fn get_src(idx: u32) -> u32 {
    return (src_buf[idx / 2] >> ((idx & 1) * 16)) & 0xFFFF;
}

fn clip_pos(x: vec2<u32>) -> vec2<u32> {
    return x & vec2(0x3FFu, 0x1FFu);
}

fn get_vram(p: vec2<u32>) -> u32 {
    let q = clip_pos(p);
    let idx = q.y * 1024u + q.x;
    return get_src(idx);
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
