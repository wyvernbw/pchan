struct VertexInput {
    @location(0) position: vec2<u32>,
    @location(1) color_and_mode: u32,
    @location(2) clut: vec2<u32>,
    @location(3) uv: vec2<u32>,
    @location(4) texpage_base: vec2<u32>,
    @location(5) textured: u32
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
};

@group(0) @binding(0)
var vram_t : texture_storage_2d<r32uint,read>;

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

    // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#clut-attribute-color-lookup-table-aka-palette
    out.clut = vec2<f32>(vec2(in.clut.x * 16, in.clut.y));

    out.uv = vec2<f32>(in.uv);
    out.textured = in.textured;
    out.texpage_base = vec2<f32>(vec2(in.texpage_base.x * 64, in.texpage_base.y * 256));

    out.color = rgb8_split_color(color);
    return out;
}

fn pack_color(color: vec3<f32>) -> u32 {
    let r = u32(color.r * 31.0);
    let g = u32(color.g * 31.0);
    let b = u32(color.b * 31.0);
    return r | (g << 5u) | (b << 10u);
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

fn read_8bit(coord: vec2<u32>) -> u32 {
    return 0;
}

fn pack_h(v: vec2<f32>, f: f32) -> vec2<f32> {
    return vec2<f32>(v.x * f, v.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    // return pack_color(vec3<f32>((in.texpage_base + vec2(in.uv.x * 4, in.uv.y)) / vec2(1024.0, 512.0), 1.0));
    // return pack_color(vec3<f32>(in.texpage_base / vec2(1024.0, 512.0), 0.0));
    // return read_16bit(vec2(512, 0) + vec2(in.uv.x, in.uv.y) / 4);
    // return read_16bit(in.texpage_base + vec2(in.uv.x, in.uv.y) / 4);
    var color = pack_color(in.color);

    switch in.color_mode {
        case COLOR_MODE_4BIT: {
            if in.textured != 0 {
                let clut_idx = read_4bit(pack_h(in.texpage_base, 4) + pack_h(in.uv, 4));
                let coord = vec2(in.clut.x + f32(clut_idx), in.clut.y);
                let clut_color = read_16bit(coord);
                color = clut_color;
                // color = clut_idx;
            }

            return color;
        }
        case COLOR_MODE_8BIT: {
            if in.textured != 0 {
                // let clut_idx = read_8bit(in.texpage_base + in.uv);
                // let coord = vec2(in.clut.x + clut_idx, in.clut.y);
                // let clut_color = read_16bit(coord);
                // color = clut_color;
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
