// https://www.w3.org/TR/WGSL

struct InstanceInput {
    @location(0) loc: vec3<f32>,
    @location(1) scale: f32,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) radius: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
	instance: InstanceInput,
) -> VertexOutput {
    // triangle strip quad
    // 0: [-1, -1]
    // 1: [ 1, -1]
    // 2: [ 1,  1]
    // 3: [-1,  1]
    // 4: [-1, -1]
    let x = f32(i32(((in_vertex_index + 1u) / 2u) & 1u) * 2 - 1);
    let y = f32(i32((in_vertex_index / 2u) & 1u) * 2 - 1);

    let uv = vec2<f32>(x, y);
    let p = vec3<f32>(uv, 0.0) * instance.scale + instance.loc;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(p, 1.0);
    out.uv = uv;
    out.color = instance.color;
    out.radius = instance.scale;
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) frag_depth: f32,
};

@fragment
fn fs_main(inp: VertexOutput) -> FragOutput {
    var out: FragOutput;
    
    let distance_to_center_squared = dot(inp.uv, inp.uv);
    if distance_to_center_squared > 1f {
        discard;
    }

    let dr = sqrt(1f - distance_to_center_squared);
    let hit_normal = normalize(vec3<f32>(inp.uv, dr));

    // the direction to the light source
    // not the direction light is traveling
    let light_direction = normalize(vec3<f32>(1f, -2f, 3f));

    let intensity = 0.2f + max(dot(light_direction, hit_normal), 0f) * 2f;

    // x * x + y * y = 1.0
    // 1.0 - x * x = y * y
    // sqrt(1.0 - x * x) = y
    
    out.color = vec4<f32>(inp.color * intensity, 1.0);
    out.frag_depth = dr * inp.radius + inp.clip_position.z;
    
    return out;
    // return vec4<f32>(base_color * intensity, 1.0);
}
