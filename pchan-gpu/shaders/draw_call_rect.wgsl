struct VertexInput {
    @location(0) position: vec2<u32>,
    @location(1) color_and_mode: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(4) color_mode: u32,
};

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

    out.clip_position = vec4<f32>(f32(in.position.x) / 512.0 - 1.0, f32(in.position.y) / 256.0 - 1.0, 0.0, 1.0);
    out.color_mode = color_mode;

    switch color_mode {
        case COLOR_MODE_15BIT, default: {
           out.color = rgb5_split_color(color);
        }
        case COLOR_MODE_24BIT: {
            out.color = rgb8_split_color(color);
        }
    }
    return out;
}

fn pack_color(color: vec3<f32>) -> u32 {
    let r = u32(color.r * 31.0);
    let g = u32(color.g * 31.0);
    let b = u32(color.b * 31.0);
    return r | (g << 5u) | (b << 10u);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    return pack_color(in.color);
    // switch in.color_mode {
    //     case COLOR_MODE_4BIT: {
    //         return pack_color(vec3(0.0, 0.0, 1.0));
    //     }
    //     case COLOR_MODE_8BIT: {
    //         return pack_color(vec3(0.0, 1.0, 0.0));
    //     }
    //     case COLOR_MODE_24BIT: {
    //         return pack_color(vec3(0.0, 1.0, 0.0));
    //     }
    //     case COLOR_MODE_15BIT, default: {
    //         return pack_color(vec3(1.0, 0.0, 0.0));
    //     }
    // }
}
