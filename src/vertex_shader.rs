pub mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        // The GPU picks each element from the vertex buffer array and calls this vertex shader program on them

        // This tells the shader that it'll receive as 'input' (hence the `in`) a `vec2` named `poop_position`
        layout(location = 0) in vec2 poop_position;

        // Called once for each vertex, placing the vertex at a position on the screen where it should be
        void main() {
            gl_Position = vec4(poop_position, 0.0, 1.0);
        }
        "
    }
}