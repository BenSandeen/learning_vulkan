#version 450

// The GPU picks each element from the vertex buffer array and calls this vertex shader program on them

// This tells the shader that it'll receive as 'input' (hence the `in`) a `vec2` named `poop_position`
layout(location = 0) in vec2 poop_position;

// Called once for each vertex, placing the vertex at a position on the screen where it should be
void main() {
    // `gl_Position` is a magic global variable.  It must be given a value for the GPU to know where to place the
    // vertex.  Only the vertex shader has access to this variable, not the fragment shader
    gl_Position = vec4(poop_position, 0.0, 1.0);
}