// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct VertexOutput {
    metal::float4 clip_position;
    metal::float3 vert_pos;
    metal::float2 uv;
    char _pad3[8];
};
struct DisplayUniforms {
    metal::uint2 display_area_pos;
    metal::uint2 resolution;
    metal::uint2 screen_rect;
    uint debug_display;
    uint srgb;
    uint color_depth;
    char _pad6[4];
};
struct type_8 {
    metal::float2 inner[6];
};

uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint read_24bit(
    metal::float2 coord,
    metal::texture2d<uint, metal::access::sample> render_t
) {
    metal::uint2 texcoord = metal::uint2(naga_f2u32(coord.x), naga_f2u32(coord.y));
    metal::uint2 texcoord_next = metal::uint2(texcoord.x + 1u, texcoord.y);
    metal::uint4 _e13 = render_t.read(metal::uint2(texcoord), 0);
    uint left = _e13.x;
    metal::uint4 _e17 = render_t.read(metal::uint2(texcoord_next), 0);
    uint right = _e17.x;
    if (naga_mod(texcoord.x, 2u) == 0u) {
        return left | ((right & 255u) << 16u);
    } else {
        return (left >> 8u) | (right << 8u);
    }
}

uint read_16bit(
    metal::float2 coord_1,
    metal::texture2d<uint, metal::access::sample> render_t
) {
    metal::uint2 texcoord_1 = metal::uint2(naga_f2u32(coord_1.x), naga_f2u32(coord_1.y));
    metal::uint4 _e8 = render_t.read(metal::uint2(texcoord_1), 0);
    return _e8.x;
}

metal::float3 rgb8_split_color(
    uint value
) {
    return metal::unpack_unorm4x8_to_float(value).xyz;
}

metal::float3 rgb5_split_color(
    uint value_1
) {
    float r = static_cast<float>(value_1 & 31u) / 31.0;
    float g = static_cast<float>((value_1 >> 5u) & 31u) / 31.0;
    float b = static_cast<float>((value_1 >> 10u) & 31u) / 31.0;
    return metal::float3(r, g, b);
}

float srgb_to_linear(
    float c
) {
    if (c <= 0.04045) {
        return c / 12.92;
    } else {
        return metal::pow((c + 0.055) / 1.055, 2.4);
    }
}

struct vs_mainInput {
};
struct vs_mainOutput {
    metal::float4 clip_position [[position]];
    metal::float3 vert_pos [[user(loc0), center_perspective]];
    metal::float2 uv [[user(loc1), center_perspective]];
};
vertex vs_mainOutput vs_main(
  uint in_vertex_index [[vertex_id]]
, constant DisplayUniforms& display [[user(fake0)]]
) {
    VertexOutput out = {};
    metal::float2 pos = {};
    type_8 vertices = type_8 {{metal::float2(-1.0, -1.0), metal::float2(-1.0, 1.0), metal::float2(1.0, 1.0), metal::float2(-1.0, -1.0), metal::float2(1.0, 1.0), metal::float2(1.0, -1.0)}};
    metal::float2 res = {};
    metal::float2 scale = {};
    uint _e26 = display.debug_display;
    if (_e26 != 0u) {
        res = metal::float2(1024.0, 512.0);
    } else {
        metal::uint2 _e34 = display.resolution;
        metal::uint2 _e37 = display.display_area_pos;
        res = static_cast<metal::float2>(_e34 - _e37);
    }
    float _e41 = res.x;
    float _e43 = res.y;
    float tex_aspect = _e41 / _e43;
    uint _e48 = display.screen_rect.x;
    uint _e53 = display.screen_rect.y;
    float screen_aspect = static_cast<float>(_e48) / static_cast<float>(_e53);
    if (tex_aspect > screen_aspect) {
        scale = metal::float2(1.0, screen_aspect / tex_aspect);
    } else {
        scale = metal::float2(tex_aspect / screen_aspect, 1.0);
    }
    metal::float2 _e65 = vertices.inner[in_vertex_index];
    pos = _e65;
    metal::float2 _e67 = pos;
    out.uv = (_e67 + metal::float2(1.0)) * metal::float2(0.5);
    metal::float2 _e74 = pos;
    metal::float2 _e75 = scale;
    pos = _e74 * _e75;
    float _e79 = pos.x;
    float _e81 = pos.y;
    out.clip_position = metal::float4(_e79, _e81, 0.0, 1.0);
    metal::float4 _e87 = out.clip_position;
    out.vert_pos = _e87.xyz;
    VertexOutput _e89 = out;
    const auto _tmp = _e89;
    return vs_mainOutput { _tmp.clip_position, _tmp.vert_pos, _tmp.uv };
}

metal::uint2 naga_f2u32(metal::float2 value) {
    return static_cast<metal::uint2>(metal::clamp(value, 0.0, 4294967000.0));
}


struct fs_mainInput {
    metal::float3 vert_pos [[user(loc0), center_perspective]];
    metal::float2 uv_1 [[user(loc1), center_perspective]];
};
struct fs_mainOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment fs_mainOutput fs_main(
  fs_mainInput varyings_1 [[stage_in]]
, metal::float4 clip_position [[position]]
, metal::texture2d<uint, metal::access::sample> render_t [[user(fake0)]]
, constant DisplayUniforms& display [[user(fake0)]]
) {
    const VertexOutput in = { clip_position, varyings_1.vert_pos, varyings_1.uv_1 };
    metal::float2 uv = {};
    bool local = {};
    bool local_1 = {};
    bool local_2 = {};
    bool local_3 = {};
    bool local_4 = {};
    bool local_5 = {};
    bool local_6 = {};
    uint col = {};
    metal::float3 out_1 = {};
    uint col_1 = {};
    metal::float3 out_2 = {};
    uv = in.uv;
    uint _e5 = display.debug_display;
    if (_e5 != 0u) {
        uv = in.uv * metal::float2(1024.0, 512.0);
        float _e15 = uv.y;
        uv.y = 512.0 - _e15;
        metal::float2 _e18 = uv;
        metal::uint2 p = naga_f2u32(_e18);
        metal::uint2 a = display.display_area_pos;
        metal::uint2 _e25 = display.resolution;
        metal::uint2 _e28 = display.display_area_pos;
        metal::uint2 b_1 = _e25 + _e28;
        if (p.x >= a.x) {
            local = p.x <= b_1.x;
        } else {
            local = false;
        }
        bool _e39 = local;
        if (_e39) {
            local_1 = p.y >= a.y;
        } else {
            local_1 = false;
        }
        bool _e46 = local_1;
        if (_e46) {
            local_2 = p.y <= b_1.y;
        } else {
            local_2 = false;
        }
        bool in_box = local_2;
        if (!((p.x == a.x))) {
            local_3 = p.x == b_1.x;
        } else {
            local_3 = true;
        }
        bool _e64 = local_3;
        if (!(_e64)) {
            local_4 = p.y == a.y;
        } else {
            local_4 = true;
        }
        bool _e72 = local_4;
        if (!(_e72)) {
            local_5 = p.y == b_1.y;
        } else {
            local_5 = true;
        }
        bool on_edge = local_5;
        if (in_box) {
            local_6 = on_edge;
        } else {
            local_6 = false;
        }
        bool _e84 = local_6;
        if (_e84) {
            return fs_mainOutput { metal::float4(1.0, 0.0, 0.0, 1.0) };
        }
    } else {
        float _e92 = uv.y;
        uv.y = 1.0 - _e92;
        metal::float2 _e95 = uv;
        metal::uint2 _e98 = display.resolution;
        metal::uint2 _e106 = display.display_area_pos;
        uv = (_e95 * static_cast<metal::float2>(_e98 - metal::uint2(2u))) + static_cast<metal::float2>(_e106);
    }
    uint _e111 = display.color_depth;
    switch(_e111) {
        case 0u:
        default: {
            metal::float2 _e112 = uv;
            uint _e113 = read_16bit(_e112, render_t);
            col = _e113;
            uint _e115 = col;
            metal::float3 _e116 = rgb5_split_color(_e115);
            out_1 = _e116;
            uint _e120 = display.srgb;
            if (_e120 != 0u) {
                float _e125 = out_1.x;
                float _e126 = srgb_to_linear(_e125);
                out_1.x = _e126;
                float _e129 = out_1.y;
                float _e130 = srgb_to_linear(_e129);
                out_1.y = _e130;
                float _e133 = out_1.z;
                float _e134 = srgb_to_linear(_e133);
                out_1.z = _e134;
            }
            metal::float3 _e135 = out_1;
            return fs_mainOutput { metal::float4(_e135, 1.0) };
        }
        case 1u: {
            metal::float2 _e138 = uv;
            uint _e139 = read_24bit(_e138, render_t);
            col_1 = _e139;
            uint _e141 = col_1;
            metal::float3 _e142 = rgb8_split_color(_e141);
            out_2 = _e142;
            uint _e146 = display.srgb;
            if (_e146 != 0u) {
                float _e151 = out_2.x;
                float _e152 = srgb_to_linear(_e151);
                out_2.x = _e152;
                float _e155 = out_2.y;
                float _e156 = srgb_to_linear(_e155);
                out_2.y = _e156;
                float _e159 = out_2.z;
                float _e160 = srgb_to_linear(_e159);
                out_2.z = _e160;
            }
            metal::float3 _e161 = out_2;
            return fs_mainOutput { metal::float4(_e161, 1.0) };
        }
    }
}
