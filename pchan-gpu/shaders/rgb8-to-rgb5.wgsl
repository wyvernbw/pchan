@group(0) @binding(0)
var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dest_buf: array<u32>;


@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2(gid.x*2, gid.y);
    if any(gid.xy >= vec2(1024, 512)) {
        return;
    }

    let left = read_raw16(vec2(p.x,p.y));
    let right = read_raw16(vec2(p.x+1, p.y));

    let idx = gid.x + gid.y*512;
    write_pair(idx, left, right);
}

fn vramcoord_to_texcoord(coord: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(vec2(coord.x/2, coord.y));
}

fn write_pair(word_idx: u32, lo: u32, hi: u32) {
    dest_buf[word_idx] = (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16u);
}

fn read_raw16(coord: vec2<u32>) -> u32 {
    let t = textureLoad(src_tex, coord, 0);
    let r = u32(round(t.r * 31.0));
    let g = u32(round(t.g * 31.0));
    let b = u32(round(t.b * 31.0));
    let a = u32(round(t.a));
    return r | (g << 5u) | (b << 10u) | (a << 15u);
}

fn clip_pos(x: vec2<u32>) -> vec2<u32> {
    return x & vec2(0x3FFu, 0x1FFu);
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
