// This macro "specifies" our shader.  Using this will let cargo compile the shader properly for us!
pub mod cs {
    vulkano_shaders::shader! {
    ty: "compute",
    src: "
    #version 450  // Version of GLSL to use

    // Set up the size of the 'work groups'.  These are chunks of work that seem to correspond to the amount of work each
    // thread will perform.  In our sample case here, since each thread gets 64 numbers to multiply by 12, there will be 1024
    // total threads used, since 64 * 1024 = 65536.  We can also use a y and z dimension if we're accessing a two or three
    // dimensional data structure.  This results in each possible combination of x, y, and z being performed (in this case,
    // 64 * 1 * 1 operations).  Work groups should always be at least size 32 to 64 for performance reasons
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    // Declares a 'descriptor' named `buf` to be used to access the data later in the code.  'Descriptors' are like handles
    // or references to various possible things, including buffers, images, sampled images, and arrays.  A descriptor is
    // necessary to be able to access and interact with the data from the CPU.  Descriptors are grouped by sets.  Thus, the
    // // `layout(set = 0, binding = 0)` attribute in the GLSL code indicates that this descriptor is the descriptor 0 in the
    // // set 0.  This declaration here does not actually declare the descriptor set, but rather just a slot for holding a
    // // descriptor set.  Before actually computing anything, we need to bind an actual descriptor to the descriptor set.
    layout(set = 0, binding = 0) buffer Data {
        uint data[];// `buf` holds an unsized array of `uint`s
    } buf;

    // Function to be invoked on each value (65536 times in total)
    void main() {
        uint idx = gl_GlobalInvocationID.x;// This global var will have a value between 0 and 65536
        buf.data[idx] *= 12;
    }"
    }
}
