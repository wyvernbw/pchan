struct Uniforms {
    kind: u32,
    src: vec2<u32>,
    dest: vec2<u32>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage> src: array<u32>;
@group(0) @binding(2) var dest: texture_storage_2d<r32uint,write>;

const CPU_TO_VRAM: u32 = 0;
const VRAM_TO_VRAM: u32 = 0;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
}
