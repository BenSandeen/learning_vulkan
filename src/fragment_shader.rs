pub mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450

        // After having run the vertex shader, the GPU figures out which pixels are inside all of the triangles that are being
        // drawn (or more precisely, which pixels' centers are inside of the triangles).  The GPU then runs the fragment shader
        // on each of those pixels

        // The fragment shader sends back as output (hence the `out`) the color of each pixel, in the form of a `vec4`
        layout(location = 0) out vec4 f_color_poop;

        void main() {
            f_color_poop = vec4(1.0, 0.0, 0.0, 1.0);  // We'll just make a red triangle for now
        }
        "
    }
}