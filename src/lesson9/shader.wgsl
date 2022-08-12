struct InstanceInput {
    @location(0) loc: vec3<f32>,
    @location(1) scale: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
	instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    // triangle strip quad
    // 0: [-1, -1]
    // 1: [ 1, -1]
    // 2: [ 1,  1]
    // 3: [-1,  1]
    // 4: [-1, -1]
    let x = f32(i32(((in_vertex_index + 1u) / 2u) & 1u) * 2 - 1);
    let y = f32(i32((in_vertex_index / 2u) & 1u) * 2 - 1);

    let p = vec3<f32>(x, y, 0.0) * instance.scale + instance.loc;

    out.clip_position = vec4<f32>(p, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.15, 0.1, 0.05, 1.0);
}
