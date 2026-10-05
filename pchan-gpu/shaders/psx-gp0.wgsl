struct VertexInput {
    @location(0) position: vec2<i32>,
    @location(1) color_and_mode: u32,
    @location(2) clut: vec2<u32>,
    @location(3) uv: vec2<u32>,
    @location(4) texpage_base: vec2<u32>,
    @location(5) flags: u32,
    @location(6) draw_area_top_left: vec2<u32>,
    @location(7) draw_area_bottom_right: vec2<u32>,
    @location(8) draw_offset: vec2<i32>,
    @location(9) gpustat: u32,
    @location(10) tex_window: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @interpolate(linear) @location(4) vram_position: vec2<f32>,
    @interpolate(flat) @location(5) color_mode: u32,
    @interpolate(flat) @location(6) clut: vec2<f32>,
    @interpolate(linear) @location(7) uv: vec2<f32>,
    @interpolate(flat) @location(8) texpage_base: vec2<f32>,
    @interpolate(flat) @location(10) flags: u32,
    @interpolate(flat) @location(11) draw_area_top_left: vec2<f32>,
    @interpolate(flat) @location(12) draw_area_bottom_right: vec2<f32>,
    @interpolate(flat) @location(13) draw_offset: vec2<f32>,
    @interpolate(flat) @location(14) gpustat: u32,
    @interpolate(flat) @location(15) tex_window: u32
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


fn rgb5_split_color(value: u32) -> vec4<f32> {
    let r = f32(value & 0x1Fu) / 31.0;
    let g = f32((value >> 5u) & 0x1Fu) / 31.0;
    let b = f32((value >> 10u) & 0x1Fu) / 31.0;
    let stp = f32(get_flag(value, 15));
    return vec4(r, g, b, stp);
}

fn rgb8_split_color(value: u32) -> vec4<f32> {
    // unpack4x8unorm unpacks as
    // 0xrrggbb
    // we need
    // 0xbbggrr (r is lsb)
    return unpack4x8unorm(value);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let color_mode = (in.color_and_mode >> 24u) & 0xFFu;
    let color = in.color_and_mode & 0x00FFFFFFu;

    let pos = vec2<i32>(vec2<i32>(in.position) + vec2(in.draw_offset.x, in.draw_offset.y));
    out.clip_position = vec4<f32>(f32(pos.x) / 512.0 - 1.0, f32(512 - pos.y) / 256.0 - 1.0, 0.0, 1.0);
    out.vram_position = vec2<f32>(vec2<i32>(in.position) + in.draw_offset);
    out.color_mode = color_mode;
    out.flags = in.flags;
    out.gpustat = in.gpustat;
    out.gpustat = in.gpustat;
    out.draw_area_top_left = vec2<f32>(in.draw_area_top_left);
    out.draw_area_bottom_right = vec2<f32>(in.draw_area_bottom_right);
    out.draw_offset = vec2<f32>(in.draw_offset);

    // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#clut-attribute-color-lookup-table-aka-palette
    out.clut = vec2<f32>(vec2(in.clut.x * 16, in.clut.y));

    out.uv = vec2<f32>(in.uv);
    out.texpage_base = vec2<f32>(vec2(in.texpage_base.x * 64, in.texpage_base.y * 256));

    out.color = rgb8_split_color(color);

    return out;
}

/// # GP0(E2h) - Texture Window setting
///
///   0-4    Texture window Mask X   (in 8 pixel steps)
///   5-9    Texture window Mask Y   (in 8 pixel steps)
///   10-14  Texture window Offset X (in 8 pixel steps)
///   15-19  Texture window Offset Y (in 8 pixel steps)
///   20-23  Not used (zero)
///   24-31  Command  (E2h)
struct TexWindow {
    mask: vec2<u32>,
    offset: vec2<u32>
}

fn unpack_tex_window(p: u32) -> TexWindow {
    var t: TexWindow;
    t.mask = vec2(extractBits(p, 0, 5), extractBits(p, 5, 5));
    t.offset = vec2(extractBits(p, 10, 5), extractBits(p, 15, 5));
    return t;
}

fn apply_tex_window(texcoord: vec2<f32>, t: TexWindow) -> vec2<f32> {
    var p = vec2<u32>(texcoord);
    p = (p & (~(t.mask * 8))) | ((t.offset & t.mask) * 8);
    return vec2<f32>(p);
}

fn pack_color(color: vec4<f32>) -> u32 {
    let r = u32(color.r * 31.0);
    let g = u32(color.g * 31.0);
    let b = u32(color.b * 31.0);
    var c =  r | (g << 5u) | (b << 10u);
    if color.a > 0.0 {
        rgb5_set_stp(c);
    }
    return c;
}

fn rgb5_set_stp(value: u32) -> u32 {
    let c = value | (1 << 15u);
    return c;
}

fn vramcoord_to_texcoord(coord: vec2<f32>) -> vec2<u32> {
    return vec2<u32>(vec2(coord.x / 2, coord.y));
}

fn read_16bit(coord: vec2<f32>) -> u32 {
    let texcoord = vramcoord_to_texcoord(coord);
    // wgpu tex coords are +Y = down
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
    let shift_amt = bit_idx * 8;

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

fn get_textured(flags: u32) -> bool {
    return get_flag(flags, 3);
}

fn wrap2(x: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let r = b - a;
    return a + (x - a) % r;
}

fn get_color(in: VertexOutput) -> u32 {
    var in_color = in.color;
    var set_mask = get_set_mask(in.flags);
    let textured = get_textured(in.flags);
    let tex_window = unpack_tex_window(in.tex_window);

    var tex_color_packed: u32;
    var tex_color: vec4<f32>;

    if textured {
        switch in.color_mode {
            case COLOR_MODE_4BIT: {
                let base = pack_h(in.texpage_base, 4);
                var texcoord = base  + in.uv;
                texcoord = apply_tex_window(texcoord, tex_window);
                var clut_idx = read_4bit(texcoord);

                var coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                tex_color_packed = clut_color;
            }
            case COLOR_MODE_8BIT: {
                let base = pack_h(in.texpage_base, 2);
                var texcoord = base  + in.uv;
                texcoord = apply_tex_window(texcoord, tex_window);
                var clut_idx = read_8bit(texcoord);

                let coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                tex_color_packed = clut_color;
            }
            case COLOR_MODE_15BIT, COLOR_MODE_24BIT, default: {
                let base = in.texpage_base;
                var texcoord = base  + in.uv;
                texcoord = apply_tex_window(texcoord, tex_window);
                tex_color_packed = read_16bit(texcoord);
            }
        }
        tex_color = rgb5_split_color(tex_color_packed);
        in_color = tex_color;
        if all(tex_color.rgb == vec3(0.0)) {
            return 0;
        }
    }

    // TODO: modulation and other effects
    
    if get_dither(in.flags) {
        in_color *= 255;
        var dither_pos: vec2<f32>;
        dither_pos = in.vram_position % 4;
        var dither_value: i32 = DITHER[u32(dither_pos.y)][u32(dither_pos.x)];
        in_color += f32(dither_value);
        in_color = clamp(in_color, vec4(0), vec4(0xff));
        in_color /= 255;
    }

    if any(in_color.rgb > vec3(1.0)) || any(in_color.rgb < vec3(0.0)) {
        return pack_color(vec4(1.0, 0.0, 0.0, 1.0));
    }

    let color_packed = pack_color(in_color);
    return color_packed;
}

const DITHER: array<array<i32, 4>, 4> = array(
    array(-4,  0, -3,  1),
    array( 2, -2,  3, -1),
    array(-3,  1, -4,  0),
    array( 3, -1,  2, -2),
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    if in.vram_position.x < in.draw_area_top_left.x
    || in.vram_position.y < in.draw_area_top_left.y
    || in.vram_position.x > in.draw_area_bottom_right.x
    || in.vram_position.y > in.draw_area_bottom_right.y {
        discard;
    }
    var draw_pixels = get_draw_pixels(in.flags);
    var pbw = get_flag(in.gpustat, 11);
    var pbc = get_flag(in.gpustat, 12);

    var current = read_16bit(in.vram_position);
    if pbc && get_mask_bit(current) {
        discard;
    }

    var color = get_color(in);
    if color == 0 {
        discard;
    }

    if pbw {
        color = rgb5_set_stp(color);
    }

    return color;
}
