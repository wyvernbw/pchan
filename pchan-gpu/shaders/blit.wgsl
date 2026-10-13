struct Uniforms {
    kind: u32,
    src: vec2<u32>,
    dest: vec2<u32>,
    size: vec2<u32>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> src_buf: array<u32>;
@group(0) @binding(2) var dest_tex: texture_storage_2d<r16uint,write>;
@group(0) @binding(3) var vram_tex: texture_storage_2d<r32uint,read_write>;

const CPU_TO_VRAM: u32 = 0;
const VRAM_TO_VRAM: u32 = 1;

// FIXME: this shader assumes size is even and it WILL break under certain conditions.
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var size = uniforms.size;
    var dest = clip_pos(vec2(gid.x*2, gid.y)+uniforms.dest);

    if (gid.x >= uniforms.size.x/2 || gid.y >= uniforms.size.y) { return; }

    switch uniforms.kind {
        case CPU_TO_VRAM {
            let px = gid.x * 2u;
            let has_second = px + 1u < size.x;

            let i0 = gid.y * size.x + px;
            let left = get_src(i0);
            var right = 0u;
            if has_second { right = get_src(i0 + 1u); }
            let packed = left | (right << 16u);

            let dest_left = dest;
            let dest_right = vec2(dest.x+1, dest.y);
            textureStore(dest_tex, dest_left, vec4(left, 0, 0, 0));
            if has_second {
                textureStore(dest_tex, dest_right, vec4(right, 0, 0, 0));
            }

            textureStore(vram_tex, vramcoord_to_texcoord(dest), vec4(packed, 0, 0, 0));
        }
        case VRAM_TO_VRAM, default: {
            let sp = vec2(gid.x * 2u, gid.y) + uniforms.src;
            let left  = get_vram(sp);
            let right = get_vram(sp + vec2(1u, 0u));
            let packed = left | (right << 16u);

            let dest_left  = vec2(dest.x,   dest.y);
            let dest_right = vec2(dest.x+1, dest.y);

            // textureStore(dest_tex, dest_left, vec4(left, 0, 0, 0));
            // textureStore(dest_tex, dest_right, vec4(right, 0, 0, 0));
            // textureStore(vram_tex, vramcoord_to_texcoord(dest), vec4(packed, 0, 0, 0));
            textureStore(dest_tex, dest_left, vec4(0xFFFF, 0, 0, 0));
            textureStore(dest_tex, dest_right, vec4(0xFFFF, 0, 0, 0));
            textureStore(vram_tex, vramcoord_to_texcoord(dest), vec4(0xFFFFFFFF, 0, 0, 0));
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
