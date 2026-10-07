// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct VertexInput {
    metal::int2 position;
    uint color_and_mode;
    char _pad2[4];
    metal::uint2 clut;
    metal::uint2 uv;
    metal::uint2 texpage_base;
    uint flags;
    char _pad6[4];
    metal::uint2 draw_area_top_left;
    metal::uint2 draw_area_bottom_right;
    metal::int2 draw_offset;
    uint gpustat;
    uint tex_window;
};
struct VertexOutput {
    metal::float4 clip_position;
    metal::float4 color;
    metal::float2 vram_position;
    uint color_mode;
    char _pad4[4];
    metal::float2 clut;
    metal::float2 uv;
    metal::float2 texpage_base;
    uint flags;
    char _pad8[4];
    metal::float2 draw_area_top_left;
    metal::float2 draw_area_bottom_right;
    metal::float2 draw_offset;
    uint gpustat;
    uint tex_window;
};
struct TexWindow {
    metal::uint2 mask;
    metal::uint2 offset;
};
struct type_9 {
    int inner[4];
};
struct type_10 {
    type_9 inner[4];
};
constant uint COLOR_MODE_4BIT = 0u;
constant uint COLOR_MODE_8BIT = 1u;
constant uint COLOR_MODE_15BIT = 2u;
constant uint COLOR_MODE_24BIT = 3u;
constant type_10 DITHER = type_10 {{type_9 {{-4, 0, -3, 1}}, type_9 {{2, -2, 3, -1}}, type_9 {{-3, 1, -4, 0}}, type_9 {{3, -1, 2, -2}}}};

bool get_flag(
    uint flags,
    uint idx
) {
    return (flags & (1u << idx)) != 0u;
}

metal::float4 rgb5_split_color(
    uint value
) {
    float r = static_cast<float>(value & 31u) / 31.0;
    float g = static_cast<float>((value >> 5u) & 31u) / 31.0;
    float b_1 = static_cast<float>((value >> 10u) & 31u) / 31.0;
    bool _e21 = get_flag(value, 15u);
    float stp = static_cast<float>(_e21);
    return metal::float4(r, g, b_1, stp);
}

metal::float4 rgb8_split_color(
    uint value_1
) {
    return metal::unpack_unorm4x8_to_float(value_1);
}

TexWindow unpack_tex_window(
    uint p
) {
    TexWindow t = {};
    t.mask = metal::uint2(metal::extract_bits(p, metal::min(0u, 32u), metal::min(5u, 32u - metal::min(0u, 32u))), metal::extract_bits(p, metal::min(5u, 32u), metal::min(5u, 32u - metal::min(5u, 32u))));
    t.offset = metal::uint2(metal::extract_bits(p, metal::min(10u, 32u), metal::min(5u, 32u - metal::min(10u, 32u))), metal::extract_bits(p, metal::min(15u, 32u), metal::min(5u, 32u - metal::min(15u, 32u))));
    TexWindow _e18 = t;
    return _e18;
}

metal::uint2 naga_f2u32(metal::float2 value) {
    return static_cast<metal::uint2>(metal::clamp(value, 0.0, 4294967000.0));
}

metal::float2 apply_tex_window(
    metal::float2 texcoord,
    TexWindow t_1
) {
    metal::uint2 p_1 = {};
    p_1 = naga_f2u32(texcoord);
    metal::uint2 _e4 = p_1;
    p_1 = (_e4 & ~((t_1.mask * 8u))) | ((t_1.offset & t_1.mask) * 8u);
    metal::uint2 _e16 = p_1;
    return static_cast<metal::float2>(_e16);
}

uint rgb5_set_stp(
    uint value_2
) {
    uint c_1 = value_2 | 32768u;
    return c_1;
}

uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}

uint pack_color(
    metal::float4 color_1
) {
    uint c = {};
    uint r_1 = naga_f2u32(color_1.x * 31.0);
    uint g_1 = naga_f2u32(color_1.y * 31.0);
    uint b_2 = naga_f2u32(color_1.z * 31.0);
    c = (r_1 | (g_1 << 5u)) | (b_2 << 10u);
    if (color_1.w > 0.0) {
        uint _e23 = c;
        uint _e24 = rgb5_set_stp(_e23);
    }
    uint _e25 = c;
    return _e25;
}

metal::uint2 vramcoord_to_texcoord(
    metal::float2 coord
) {
    return naga_f2u32(metal::float2(coord.x / 2.0, coord.y));
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint read_16bit(
    metal::float2 coord_1,
    metal::texture2d<uint, metal::access::read> vram_t
) {
    uint packed = {};
    metal::uint2 _e1 = vramcoord_to_texcoord(coord_1);
    metal::uint4 _e3 = vram_t.read(metal::uint2(_e1));
    packed = _e3.x;
    uint _e6 = packed;
    return (_e6 >> (naga_mod(naga_f2u32(coord_1.x), 2u) * 16u)) & 65535u;
}

uint read_4bit(
    metal::float2 coord_2,
    metal::texture2d<uint, metal::access::read> vram_t
) {
    uint packed_1 = {};
    uint _e6 = read_16bit(metal::float2(coord_2.x / 4.0, coord_2.y), vram_t);
    packed_1 = _e6;
    uint bit_idx = naga_mod(naga_f2u32(coord_2.x), 4u);
    uint shift_amt = bit_idx * 4u;
    uint _e14 = packed_1;
    return (_e14 >> shift_amt) & 15u;
}

uint read_8bit(
    metal::float2 coord_3,
    metal::texture2d<uint, metal::access::read> vram_t
) {
    uint packed_2 = {};
    uint _e6 = read_16bit(metal::float2(coord_3.x / 2.0, coord_3.y), vram_t);
    packed_2 = _e6;
    uint bit_idx_1 = naga_mod(naga_f2u32(coord_3.x), 2u);
    uint shift_amt_1 = bit_idx_1 * 8u;
    uint _e14 = packed_2;
    return (_e14 >> shift_amt_1) & 255u;
}

metal::float2 pack_h(
    metal::float2 v,
    float f
) {
    return metal::float2(v.x * f, v.y);
}

bool get_mask_bit(
    uint color_2
) {
    bool _e2 = get_flag(color_2, 15u);
    return _e2;
}

bool get_dither(
    uint flags_1
) {
    bool _e2 = get_flag(flags_1, 0u);
    return _e2;
}

bool get_set_mask(
    uint flags_2
) {
    bool _e2 = get_flag(flags_2, 1u);
    return _e2;
}

bool get_draw_pixels(
    uint flags_3
) {
    bool _e2 = get_flag(flags_3, 2u);
    return _e2;
}

bool get_textured(
    uint flags_4
) {
    bool _e2 = get_flag(flags_4, 3u);
    return _e2;
}

metal::float2 wrap2_(
    metal::float2 x,
    metal::float2 a,
    metal::float2 b
) {
    metal::float2 r_2 = b - a;
    return a + metal::fmod(x - a, r_2);
}

uint get_color(
    VertexOutput in_2,
    metal::texture2d<uint, metal::access::read> vram_t
) {
    metal::float4 in_color = {};
    bool set_mask = {};
    uint tex_color_packed = {};
    metal::float4 tex_color = {};
    metal::float2 texcoord_1 = {};
    uint clut_idx = {};
    metal::float2 coord_4 = {};
    metal::float2 texcoord_2 = {};
    uint clut_idx_1 = {};
    metal::float2 texcoord_3 = {};
    metal::float2 dither_pos = {};
    int dither_value = {};
    bool local_4 = {};
    in_color = in_2.color;
    bool _e4 = get_set_mask(in_2.flags);
    set_mask = _e4;
    bool _e7 = get_textured(in_2.flags);
    TexWindow _e9 = unpack_tex_window(in_2.tex_window);
    if (_e7) {
        switch(in_2.color_mode) {
            case 0u: {
                metal::float2 _e15 = pack_h(in_2.texpage_base, 4.0);
                texcoord_1 = _e15 + in_2.uv;
                metal::float2 _e19 = texcoord_1;
                metal::float2 _e20 = apply_tex_window(_e19, _e9);
                texcoord_1 = _e20;
                metal::float2 _e21 = texcoord_1;
                uint _e22 = read_4bit(_e21, vram_t);
                clut_idx = _e22;
                uint _e26 = clut_idx;
                coord_4 = metal::float2(in_2.clut.x + static_cast<float>(_e26), in_2.clut.y);
                metal::float2 _e33 = coord_4;
                uint _e34 = read_16bit(_e33, vram_t);
                tex_color_packed = _e34;
                break;
            }
            case 1u: {
                metal::float2 _e37 = pack_h(in_2.texpage_base, 2.0);
                texcoord_2 = _e37 + in_2.uv;
                metal::float2 _e41 = texcoord_2;
                metal::float2 _e42 = apply_tex_window(_e41, _e9);
                texcoord_2 = _e42;
                metal::float2 _e43 = texcoord_2;
                uint _e44 = read_8bit(_e43, vram_t);
                clut_idx_1 = _e44;
                uint _e48 = clut_idx_1;
                metal::float2 coord_5 = metal::float2(in_2.clut.x + static_cast<float>(_e48), in_2.clut.y);
                uint _e54 = read_16bit(coord_5, vram_t);
                tex_color_packed = _e54;
                break;
            }
            case 2u:
            case 3u:
            default: {
                metal::float2 base = in_2.texpage_base;
                texcoord_3 = base + in_2.uv;
                metal::float2 _e59 = texcoord_3;
                metal::float2 _e60 = apply_tex_window(_e59, _e9);
                texcoord_3 = _e60;
                metal::float2 _e61 = texcoord_3;
                uint _e62 = read_16bit(_e61, vram_t);
                tex_color_packed = _e62;
                break;
            }
        }
        uint _e63 = tex_color_packed;
        metal::float4 _e64 = rgb5_split_color(_e63);
        tex_color = _e64;
        metal::float4 _e65 = tex_color;
        in_color = _e65;
        metal::float4 _e66 = tex_color;
        if (metal::all(_e66.xyz == metal::float3(0.0))) {
            return 0u;
        }
    }
    bool _e74 = get_dither(in_2.flags);
    if (_e74) {
        metal::float4 _e75 = in_color;
        in_color = _e75 * 255.0;
        dither_pos = metal::fmod(in_2.vram_position, metal::float2(4.0));
        float _e85 = dither_pos.y;
        float _e89 = dither_pos.x;
        dither_value = DITHER.inner[naga_f2u32(_e85)].inner[naga_f2u32(_e89)];
        metal::float4 _e93 = in_color;
        int _e94 = dither_value;
        in_color = _e93 + metal::float4(static_cast<float>(_e94));
        metal::float4 _e98 = in_color;
        in_color = metal::clamp(_e98, metal::float4(0.0), metal::float4(255.0));
        metal::float4 _e104 = in_color;
        in_color = _e104 / metal::float4(255.0);
    }
    metal::float4 _e108 = in_color;
    if (!(metal::any(_e108.xyz > metal::float3(1.0)))) {
        metal::float4 _e117 = in_color;
        local_4 = metal::any(_e117.xyz < metal::float3(0.0));
    } else {
        local_4 = true;
    }
    bool _e124 = local_4;
    if (_e124) {
        uint _e130 = pack_color(metal::float4(1.0, 0.0, 0.0, 1.0));
        return _e130;
    }
    metal::float4 _e131 = in_color;
    uint _e132 = pack_color(_e131);
    return _e132;
}

struct vs_mainInput {
    metal::int2 position [[attribute(0)]];
    uint color_and_mode [[attribute(1)]];
    metal::uint2 clut [[attribute(2)]];
    metal::uint2 uv [[attribute(3)]];
    metal::uint2 texpage_base [[attribute(4)]];
    uint flags_5 [[attribute(5)]];
    metal::uint2 draw_area_top_left [[attribute(6)]];
    metal::uint2 draw_area_bottom_right [[attribute(7)]];
    metal::int2 draw_offset [[attribute(8)]];
    uint gpustat [[attribute(9)]];
    uint tex_window [[attribute(10)]];
};
struct vs_mainOutput {
    metal::float4 clip_position [[position]];
    metal::float4 color [[user(loc0), center_perspective]];
    metal::float2 vram_position [[user(loc4), center_no_perspective]];
    uint color_mode [[user(loc5), flat]];
    metal::float2 clut [[user(loc6), flat]];
    metal::float2 uv [[user(loc7), center_no_perspective]];
    metal::float2 texpage_base [[user(loc8), flat]];
    uint flags [[user(loc10), flat]];
    metal::float2 draw_area_top_left [[user(loc11), flat]];
    metal::float2 draw_area_bottom_right [[user(loc12), flat]];
    metal::float2 draw_offset [[user(loc13), flat]];
    uint gpustat [[user(loc14), flat]];
    uint tex_window [[user(loc15), flat]];
};
vertex vs_mainOutput vs_main(
  vs_mainInput varyings [[stage_in]]
) {
    const VertexInput in = { varyings.position, varyings.color_and_mode, {}, varyings.clut, varyings.uv, varyings.texpage_base, varyings.flags_5, {}, varyings.draw_area_top_left, varyings.draw_area_bottom_right, varyings.draw_offset, varyings.gpustat, varyings.tex_window };
    VertexOutput out = {};
    uint color_mode = (in.color_and_mode >> 24u) & 255u;
    uint color_3 = in.color_and_mode & 16777215u;
    metal::int2 pos = static_cast<metal::int2>(as_type<metal::int2>(as_type<metal::uint2>(static_cast<metal::int2>(in.position)) + as_type<metal::uint2>(metal::int2(in.draw_offset.x, in.draw_offset.y))));
    out.clip_position = metal::float4((static_cast<float>(pos.x) / 512.0) - 1.0, (static_cast<float>(as_type<int>(as_type<uint>(512) - as_type<uint>(pos.y))) / 256.0) - 1.0, 0.0, 1.0);
    out.vram_position = static_cast<metal::float2>(as_type<metal::int2>(as_type<metal::uint2>(static_cast<metal::int2>(in.position)) + as_type<metal::uint2>(in.draw_offset)));
    out.color_mode = color_mode;
    out.flags = in.flags;
    out.gpustat = in.gpustat;
    out.gpustat = in.gpustat;
    out.draw_area_top_left = static_cast<metal::float2>(in.draw_area_top_left);
    out.draw_area_bottom_right = static_cast<metal::float2>(in.draw_area_bottom_right);
    out.draw_offset = static_cast<metal::float2>(in.draw_offset);
    out.clut = static_cast<metal::float2>(metal::uint2(in.clut.x * 16u, in.clut.y));
    out.uv = static_cast<metal::float2>(in.uv);
    out.texpage_base = static_cast<metal::float2>(metal::uint2(in.texpage_base.x * 64u, in.texpage_base.y * 256u));
    metal::float4 _e83 = rgb8_split_color(color_3);
    out.color = _e83;
    VertexOutput _e84 = out;
    const auto _tmp = _e84;
    return vs_mainOutput { _tmp.clip_position, _tmp.color, _tmp.vram_position, _tmp.color_mode, _tmp.clut, _tmp.uv, _tmp.texpage_base, _tmp.flags, _tmp.draw_area_top_left, _tmp.draw_area_bottom_right, _tmp.draw_offset, _tmp.gpustat, _tmp.tex_window };
}


struct fs_mainInput {
    metal::float4 color_3 [[user(loc0), center_perspective]];
    metal::float2 vram_position [[user(loc4), center_no_perspective]];
    uint color_mode [[user(loc5), flat]];
    metal::float2 clut [[user(loc6), flat]];
    metal::float2 uv [[user(loc7), center_no_perspective]];
    metal::float2 texpage_base [[user(loc8), flat]];
    uint flags_5 [[user(loc10), flat]];
    metal::float2 draw_area_top_left [[user(loc11), flat]];
    metal::float2 draw_area_bottom_right [[user(loc12), flat]];
    metal::float2 draw_offset [[user(loc13), flat]];
    uint gpustat [[user(loc14), flat]];
    uint tex_window [[user(loc15), flat]];
};
struct fs_mainOutput {
    uint member_1 [[color(0)]];
};
fragment fs_mainOutput fs_main(
  fs_mainInput varyings_1 [[stage_in]]
, metal::float4 clip_position [[position]]
, metal::texture2d<uint, metal::access::read> vram_t [[user(fake0)]]
) {
    const VertexOutput in_1 = { clip_position, varyings_1.color_3, varyings_1.vram_position, varyings_1.color_mode, {}, varyings_1.clut, varyings_1.uv, varyings_1.texpage_base, varyings_1.flags_5, {}, varyings_1.draw_area_top_left, varyings_1.draw_area_bottom_right, varyings_1.draw_offset, varyings_1.gpustat, varyings_1.tex_window };
    bool local = {};
    bool local_1 = {};
    bool local_2 = {};
    bool draw_pixels = {};
    bool pbw = {};
    bool pbc = {};
    uint current = {};
    bool local_3 = {};
    uint color = {};
    if (!((in_1.vram_position.x < in_1.draw_area_top_left.x))) {
        local = in_1.vram_position.y < in_1.draw_area_top_left.y;
    } else {
        local = true;
    }
    bool _e15 = local;
    if (!(_e15)) {
        local_1 = in_1.vram_position.x > in_1.draw_area_bottom_right.x;
    } else {
        local_1 = true;
    }
    bool _e25 = local_1;
    if (!(_e25)) {
        local_2 = in_1.vram_position.y > in_1.draw_area_bottom_right.y;
    } else {
        local_2 = true;
    }
    bool _e35 = local_2;
    if (_e35) {
        metal::discard_fragment();
    }
    bool _e37 = get_draw_pixels(in_1.flags);
    draw_pixels = _e37;
    bool _e41 = get_flag(in_1.gpustat, 11u);
    pbw = _e41;
    bool _e45 = get_flag(in_1.gpustat, 12u);
    pbc = _e45;
    uint _e48 = read_16bit(in_1.vram_position, vram_t);
    current = _e48;
    bool _e50 = pbc;
    if (_e50) {
        uint _e53 = current;
        bool _e54 = get_mask_bit(_e53);
        local_3 = _e54;
    } else {
        local_3 = false;
    }
    bool _e56 = local_3;
    if (_e56) {
        metal::discard_fragment();
    }
    uint _e57 = get_color(in_1, vram_t);
    color = _e57;
    uint _e59 = color;
    if (_e59 == 0u) {
        metal::discard_fragment();
    }
    bool _e62 = pbw;
    if (_e62) {
        uint _e63 = color;
        uint _e64 = rgb5_set_stp(_e63);
        color = _e64;
    }
    uint _e65 = color;
    return fs_mainOutput { _e65 };
}
