struct VertexInput {
    @location(0) position: vec2<u32>,
    @location(1) color_and_mode: u32,
    @location(2) clut: vec2<u32>,
    @location(3) uv: vec2<u32>,
    @location(4) texpage_base: vec2<u32>,
    @location(5) textured: u32,
    @location(6) flags: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @interpolate(linear) @location(4) vram_position: vec2<f32>,
    @interpolate(flat) @location(5) color_mode: u32,
    @interpolate(flat) @location(6) clut: vec2<f32>,
    @interpolate(linear) @location(7) uv: vec2<f32>,
    @interpolate(flat) @location(8) texpage_base: vec2<f32>,
    @interpolate(flat) @location(9) textured: u32,
    @interpolate(flat) @location(10) flags: u32,
};

@group(0) @binding(0)
var vram_t : texture_storage_2d<r32uint,read>;

// struct RenderUniforms {
// }

// @group(0) @binding(1)
// var<uniform> render_uniforms: RenderUniforms;

const COLOR_MODE_4BIT:  u32 = 0x0;
const COLOR_MODE_8BIT:  u32 = 0x1;
const COLOR_MODE_15BIT: u32 = 0x2;
const COLOR_MODE_24BIT: u32 = 0x3;


fn rgb5_split_color(value: u32) -> vec3<f32> {
    let r = f32(value & 0x1Fu) / 31.0;
    let g = f32((value >> 5u) & 0x1Fu) / 31.0;
    let b = f32((value >> 10u) & 0x1Fu) / 31.0;
    return vec3(r, g, b);
}

fn rgb8_split_color(value: u32) -> vec3<f32> {
    // unpack4x8unorm unpacks as
    // 0xrrggbb
    // we need
    // 0xbbggrr (r is lsb)
    return unpack4x8unorm(value).rgb;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let color_mode = (in.color_and_mode >> 24u) & 0xFFu;
    let color = in.color_and_mode & 0x00FFFFFFu;

    out.clip_position = vec4<f32>(f32(in.position.x) / 512.0 - 1.0, f32(512 - in.position.y) / 256.0 - 1.0, 0.0, 1.0);
    out.vram_position = vec2<f32>(vec2(in.position.x, 512 - in.position.y));
    out.color_mode = color_mode;
    out.flags = in.flags;

    // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#clut-attribute-color-lookup-table-aka-palette
    out.clut = vec2<f32>(vec2(in.clut.x * 16, in.clut.y));

    out.uv = vec2<f32>(in.uv);
    out.textured = in.textured;
    out.texpage_base = vec2<f32>(vec2(in.texpage_base.x * 64, in.texpage_base.y * 256));

    out.color = rgb8_split_color(color);

    return out;
}

fn pack_color(color: vec3<f32>, mask: bool) -> u32 {
    let r = u32(color.r * 31.0);
    let g = u32(color.g * 31.0);
    let b = u32(color.b * 31.0);
    var c =  r | (g << 5u) | (b << 10u);
    if mask {
        c |= (1 << 15u);
    }
    return c;
}

fn vramcoord_to_texcoord(coord: vec2<f32>) -> vec2<u32> {
    return vec2<u32>(vec2(coord.x / 2, coord.y));
}

fn read_16bit(coord: vec2<f32>) -> u32 {
    let texcoord = vramcoord_to_texcoord(coord);
    var packed = textureLoad(vram_t, texcoord).r;
    return (packed >> ((u32(coord.x) % 2) * 16)) & 0xFFFF;
}

fn read_4bit(coord: vec2<f32>) -> u32 {
    var packed = read_16bit(vec2(coord.x / 4, coord.y));
    let bit_idx = u32(coord.x) % 4;
    let shift_amt = bit_idx * 4;

    return (packed >> shift_amt) & 0xFu;
}

fn read_8bit(coord: vec2<f32>) -> u32 {
    var packed = read_16bit(vec2(coord.x / 2, coord.y));
    let bit_idx = u32(coord.x) % 2;
    let shift_amt = bit_idx * 2;

    return (packed >> shift_amt) & 0xFFu;
}

fn pack_h(v: vec2<f32>, f: f32) -> vec2<f32> {
    return vec2<f32>(v.x * f, v.y);
}

fn get_mask_bit(color: u32) -> bool {
    return get_flag(color, 15);
}

fn get_flag(flags: u32, idx: u32) -> bool {
    return (flags & (u32(1) << idx)) != 0;
}

fn get_dither(flags: u32) -> bool {
    return get_flag(flags, 0);
}

fn get_set_mask(flags: u32) -> bool {
    return get_flag(flags, 1);
}

fn get_draw_pixels(flags: u32) -> bool {
    return get_flag(flags, 2);
}

fn get_color(in: VertexOutput) -> u32 {
    var in_color = in.color;
    var set_mask = get_set_mask(in.flags);

    if get_dither(in.flags) {
        in_color *= 255;
        var dither_pos: vec2<f32>;
        dither_pos = in.vram_position % 4;
        var dither_value: i32 = DITHER[u32(dither_pos.y)][u32(dither_pos.x)];
        in_color += f32(dither_value);
        in_color = clamp(in_color, vec3(0), vec3(0xff));
        in_color /= 255;
    }

    var color = pack_color(in_color, set_mask);
    switch in.color_mode {
        case COLOR_MODE_4BIT: {
            if in.textured != 0 {
                let clut_idx = read_4bit(pack_h(in.texpage_base, 4) + pack_h(in.uv, 1));
                let coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                color = clut_color;
                // color = clut_idx;
            }

            return color;
        }
        case COLOR_MODE_8BIT: {
            if in.textured != 0 {
                let clut_idx = read_8bit(pack_h(in.texpage_base, 2) + pack_h(in.uv, 1));
                let coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                color = clut_color;
            }

            return color;
        }
        case COLOR_MODE_15BIT, COLOR_MODE_24BIT, default: {
            if in.textured != 0 {
                color = read_16bit(in.texpage_base + in.uv);
            }

            return color;
        }
    }
}

const DITHER: array<array<i32, 4>, 4> = array(
    array(-4,  0, -3,  1),
    array( 2, -2,  3, -1),
    array(-3,  1, -4,  0),
    array( 3, -1,  2, -2),
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    var draw_pixels = get_draw_pixels(in.flags);
    if draw_pixels {
        var current = read_16bit(in.vram_position);
        if get_mask_bit(current) {
            discard;
        }
    }
    var color = get_color(in);
    if color == 0 {
        discard;
    }

    return color;
}
