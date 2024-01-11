

// The input buffer, with u8 packing.
// two dimensional dynamic sizing is not possible.
@group(0) @binding(0) 
var<storage, read> input_buffer : array<u32>;

// The kernel
@group(0) @binding(1)
var<storage, read> kernel : array<f32>;

// .x is height of the input texture.
// .y is width of the texture.
// .z is the dimensions of the kernel.
@group(0) @binding(2)
var<storage, read> input_meta_data : vec3<u32>;

@group(0) @binding(3)
var<storage, read_write> output_buffer : array<u32>;  



// Compute shader which calculates the convolution,
@compute 
@workgroup_size(8, 8, 1)
fn convolute(@builtin(global_invocation_id) id : vec3<u32>) {
    
}

@compute 
@workgroup_size(8, 8, 1)
fn main() {
    
}