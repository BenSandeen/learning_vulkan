#version 450

// Each thread takes an 8x8 = 64 pixel chunk of the image
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Specify that we'll be using an image.  It's in descriptor set 0, slot 0 (binding).  `uniform` makes it basically a
// global variable, so that all threads will be working on the same image.  The `rgba8` tells it what format the image
// uses.
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
    // Remember, the `gl_GlobalInvocationID.xy` indicates which pixel the thread is working on.  We get the normalized
    // coords in order to do the math and get normalized values for the pixel.
    vec2 norm_coords = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));

    // For added obfuscation, this example code uses multiple vec2s instead of vec3s or vec4s to handle the complex
    // numbers of the Mandelbrot set
    vec2 c = (norm_coords - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(0.0, 0.0);
    float i;

    // All the stuff between here and the initialization of the vector `to_write` is just the math of the Mandelbrot
    // set.  It's correct.  Don't worry your pretty little head over understanding it when you're already trying to
    // learn Vulkan and GLSL :P
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    // Create pixel value with alpha included
    vec4 to_write = vec4(vec3(i), 1.0);

    // Store the value of the pixel `to_write` in the image `img` at the pixel coordinate `gl_GlobalInvocationID.xy`
    // (note that is first converted to an integer vector)
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}