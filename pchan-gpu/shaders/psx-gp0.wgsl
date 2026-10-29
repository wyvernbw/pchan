enable dual_source_blending;

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

struct OutColor {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) src1: vec4<f32>,
}

@group(0) @binding(0)
var vram_t : texture_storage_2d<rgba8unorm,read>;

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
    out.vram_position = vec2<f32>(pos);
    out.color_mode = color_mode;
    out.flags = in.flags;
    out.gpustat = in.gpustat;
    out.gpustat = in.gpustat;
    out.draw_area_top_left = vec2<f32>(in.draw_area_top_left);
    out.draw_area_bottom_right = vec2<f32>(in.draw_area_bottom_right);
    out.draw_offset = vec2<f32>(in.draw_offset);

    // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#clut-attribute-color-lookup-table-aka-palette
    out.clut = vec2<f32>(vec2((in.clut.x * 16) & 1023, in.clut.y));

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
   var p = vec2<u32>(max(texcoord, vec2(0.0))) & vec2(0xFFu);
    p = (p & (~(t.mask * 8))) | ((t.offset & t.mask) * 8);
    return vec2<f32>(p);
}

fn pack_color(color: vec4<f32>) -> u32 {
    let r = u32(color.r * 31.0);
    let g = u32(color.g * 31.0);
    let b = u32(color.b * 31.0);
    var c =  r | (g << 5u) | (b << 10u);
    if color.a > 0.0 {
        c = rgb5_set_stp(c);
    }
    return c;
}

fn rgb5_set_stp(value: u32) -> u32 {
    let c = value | (1 << 15u);
    return c;
}

fn vramcoord_to_texcoord(coord: vec2<f32>) -> vec2<u32> {
    return vec2<u32>(vec2(coord.x, coord.y) % vec2(1024, 512));
}

fn read_16bit(coord: vec2<f32>) -> vec4<f32> {
    let texcoord = vramcoord_to_texcoord(coord);
    // wgpu tex coords are +Y = down
    let texel = textureLoad(vram_t, texcoord);
    return texel;
}

fn read_raw16(coord: vec2<f32>) -> u32 {
    let t = read_16bit(coord);
    let r = u32(round(t.r * 31.0));
    let g = u32(round(t.g * 31.0));
    let b = u32(round(t.b * 31.0));
    let a = u32(round(t.a));
    return r | (g << 5u) | (b << 10u) | (a << 15u);
}

fn read_4bit(coord: vec2<f32>) -> u32 {
    let x = i32(floor(coord.x));
    let hw = read_raw16(vec2<f32>(f32(x >> 2u), floor(coord.y)));
    let shift = u32(x & 3) * 4u;
    return (hw >> shift) & 0xFu;
}

fn read_8bit(coord: vec2<f32>) -> u32 {
    let x = i32(floor(coord.x));
    let hw = read_raw16(vec2<f32>(f32(x >> 1u), floor(coord.y)));
    let shift = u32(x & 1) * 8u;
    return (hw >> shift) & 0xFFu;
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

fn get_textured(flags: u32) -> bool {
    return get_flag(flags, 3);
}

fn get_modulation(flags: u32) -> bool {
   return extractBits(flags, 4, 1) != 0;
}

fn get_semi_transparency(flags: u32) -> u32 {
   return extractBits(flags, 5, 2);
}

fn get_stp_on(flags: u32) -> bool {
   return extractBits(flags, 7, 1) != 0;
}

fn wrap2(x: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let r = b - a;
    return a + (x - a) % r;
}

struct Color {
    out: OutColor,
    transparent: bool
}

fn get_color(in: VertexOutput) -> Color {
    var in_color = OutColor(in.color, vec4(0.0));
    var set_mask = get_set_mask(in.flags);
    let textured = get_textured(in.flags);
    let tex_window = unpack_tex_window(in.tex_window);

    var tex_color_packed: u32;
    var tex_color: vec4<f32>;
    var has_stp = true;

    if textured {
        switch in.color_mode {
            case COLOR_MODE_4BIT: {
                let base = pack_h(in.texpage_base, 4);
                let uv = apply_tex_window(in.uv, tex_window);
                var texcoord = base  + uv;
                var clut_idx = read_4bit(texcoord) & 15;

                var coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                tex_color  = clut_color;
            }
            case COLOR_MODE_8BIT: {
                let base = pack_h(in.texpage_base, 2);
                let uv = apply_tex_window(in.uv, tex_window);
                var texcoord = base  + uv;
                var clut_idx = read_8bit(texcoord) & 255;

                let coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                tex_color = clut_color;
            }
            case COLOR_MODE_15BIT, COLOR_MODE_24BIT, default: {
                let base = in.texpage_base;
                let uv = apply_tex_window(in.uv, tex_window);
                var texcoord = base  + uv;
                tex_color = read_16bit(texcoord);
            }
        }

        if all(tex_color == vec4(0.0)) {
            return Color(OutColor(vec4(0), vec4(1)), true);
        }

        if get_modulation(in.flags) {
            in_color.color = tex_color* in.color* 2.0;
        } else {
            in_color.color = tex_color;
        }

        in_color.color.a = tex_color.a;
        has_stp = in_color.color.a > 0.5;

    } else {
        in_color.color.a = 0.0;
    }

    has_stp = has_stp && get_stp_on(in.flags);

    if get_stp_on(in.flags) {
        // A: Bx(1-f) + F
        // S: B - F
        // SO: Bx(1-f) + Fxf
        switch get_semi_transparency(in.flags) {
            // average: A
            case 0, default: {
                in_color.src1 = vec4(select(1.0, 0.5, has_stp));
                in_color.color = select(in_color.color, in_color.color * 0.5, has_stp);
            }
            // full add: A
            case 1: {
                in_color.src1 = vec4(select(1., 0., has_stp));
            }
            // subtract (S or SO)
            case 2: {
                // return Color(OutColor(vec4(1.0, 0.0, 1.0, 1.0), vec4(1)), false);
                in_color.src1 = vec4(f32(!has_stp));
            }
            // 1/4 add: A
            case 3: {
                in_color.src1 = vec4(select(1., 0., has_stp));
                in_color.color = select(in_color.color, in_color.color * 0.25, has_stp);
            }
        }
    } else {
        in_color.src1 = vec4(1);
    }
    
    if get_dither(in.flags) {
        var rgb8 = in_color.color.rgb;
        rgb8 *= 255;
        var dither_pos: vec2<f32>;
        dither_pos = in.vram_position % 4;
        var dither_value: i32 = DITHER[u32(dither_pos.y)][u32(dither_pos.x)];
        rgb8 += f32(dither_value);
        rgb8 = clamp(rgb8, vec3(0), vec3(0xff));
        rgb8 /= 255;
        in_color.color = vec4(rgb8, in_color.color.a);
    }

    return Color(in_color, false);
}

const DITHER: array<array<i32, 4>, 4> = array(
    array(-4,  0, -3,  1),
    array( 2, -2,  3, -1),
    array(-3,  1, -4,  0),
    array( 3, -1,  2, -2),
);

@fragment
fn fs_main(in: VertexOutput) -> OutColor {
    let p = floor(in.vram_position);
    if p.x < in.draw_area_top_left.x
    || p.y < in.draw_area_top_left.y
    || p.x > in.draw_area_bottom_right.x
    || p.y > in.draw_area_bottom_right.y {
        discard;
    }
    var pbw = get_flag(in.gpustat, 11);
    var pbc = get_flag(in.gpustat, 12);
    let fill = get_flag(in.flags, 2);

    if fill {
        return OutColor(in.color, vec4(1.0));
    }

    var current = read_16bit(in.vram_position);
    if pbc && current.a > 0.0 {
        discard;
    }

    var color = get_color(in);
    if color.transparent {
        discard;
    }

    if pbw && !fill {
        color.out.color.a = 1.0;
    }

    return color.out;
}
