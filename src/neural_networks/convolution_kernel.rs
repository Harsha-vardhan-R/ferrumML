//! This file contains the trait definition for the convolution and pooling and also contains the 
//! implementation for some common convolution kernels and pooling methods, 
//! user can create their own kernel by passing the matrix as an argument into the `kernel` function.
//! 
//! we use the wgpu api calls to make this part to automatically run on the respective gpu.
//! code which runs only on cpu will be implemented after some time.



use std::{f32::consts::PI, fmt::{Debug, write}};

use super::neural_network::functionValueAt;


#[derive(Clone)]
/// The kernel type, 
/// first field stores the kernel in a flat vector,
/// second field stores the dimensions,
/// third field stores the sum of all the values in the kernel matrix, used to divide for normalization in the final convolution process. 
pub struct Kernel(Vec<f32>, u32, f32);

impl Debug for Kernel {
    
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel :\n");

        for side in 0..self.1 {
            write!(f, "    ");
            for value in 0..self.1 {
                write!(f , "{:3}    ", self.0[((side*self.1)+value) as usize]);
            }
            write!(f, "\n\n");
        }

        write!(f, "kernel dimension : {}\nTo divide with after convolution : {}", self.1, self.2)
    }
    
}


/// kernel_matrix : this is the matrix that will be used to generate the kernel,
/// You can only create odd-square sized kernel, IF you strictly want a even sized or rectangular kernel,
/// just fill the kernel to the top left portion, nad fill the remaining space with 0.
pub fn kernel(kernel_matrix : &Vec<f32>) -> Kernel {
    //make sure all the rows have the same width.
    let width = kernel_matrix.len();

    if (width % 2 == 0 || !(f32::sqrt(width as f32)%1.0 < 2.0*f32::EPSILON) ) {
        panic!("Bad value for the kernel, even dimensions are not allowed, or rows or columns incomplete");
    }

    let mut normal = kernel_matrix.iter().sum();

    Kernel(kernel_matrix.clone(), width as u32, normal)
}



/// The built-in Convolutional Kernels , contains the most commonly used kernels in different fields of
/// computing.
/// These 'ConvolutionalKernel' types also have a field called size, which is going to set the size of the convolutionalKernel
/// as ALL the kernels that are prebuilt ARE square kernels the size defines the dimensions of the kernel,
/// a size value of one means the kernel matrix is only a single element,
/// for bigger sizes the library automatically calculates the values for each of the places in the 
pub enum ConvolutionKernelBuilder {
    Identity,
    /// need to initialize with the sigma value, which will be used to build the kernel.
    GaussianBlur(f32),
    BoxBlur,
    /// need to provide the value at the center, generally 4 or 8 is used.
    LaplaceSharpen(f32),
    UnsharpMask,
    HorizontalPrewitt,
    VerticalPrewitt,
    HorizontalSobel,
    VerticalSobel,
}


impl ConvolutionKernelBuilder {

    /// Use this method on the particular type of built-in `ConvolutionalKernel` type,
    /// - Pass the dimensions you need for the kernel as an `u8` argument.
    ///     * cannot have dimensions for the kernel larger than 255.
    /// - Only Odd dimensions are allowed, panics for even dimensions.
    pub fn build_with_dimen(self, dimensions : u8) -> Kernel {

        if dimensions % 2 == 0 {
            panic!("Only odd dimesions are allowed, {} is not odd", dimensions);
        }

        let mut out_vec: Vec<f32>;
        let mut normal = 0.0_f32;

        // Building the kernels based on dimensions.
        match self {
            ConvolutionKernelBuilder::Identity => {
                out_vec = vec![0.0 ; dimensions as usize*dimensions as usize];
                out_vec[(dimensions as usize*dimensions as usize)/2] = 1.0;
                normal += 1.0;
            },
            ConvolutionKernelBuilder::GaussianBlur(sigma) => {
                out_vec = vec![0.0 ; dimensions as usize*dimensions as usize];
                let half: i32 = (dimensions / 2).into();
                for i in 0..dimensions {
                    for j in 0..dimensions {
                        let temp = f32::exp(-(((i as i32 - half) as f32).powf(2.0) + ((j as i32 - half) as f32).powf(2.0))/(2.0*sigma.powf(2.0)))/(2.0*PI*sigma.powf(2.0));
                        out_vec[( i as usize *dimensions as usize)+j as usize] = temp;
                        normal += temp;
                    }
                }
            },
            ConvolutionKernelBuilder::BoxBlur => {
                out_vec = vec![1.0 ; dimensions as usize*dimensions as usize];
                normal = (dimensions * dimensions) as f32;
            },
            ConvolutionKernelBuilder::LaplaceSharpen(center) => {
                out_vec = vec![0.0 ; dimensions as usize*dimensions as usize];
                for i in 0..=dimensions/2 {
                    for (ind, j) in ((dimensions/2)-i..=(dimensions/2)).enumerate() {
                        let num = -(2.0_f32.powf(ind as f32));
                        out_vec[( i as usize *dimensions as usize)+j as usize] = num;
                        out_vec[((dimensions - i - 1) as usize*dimensions as usize)+j as usize] = num;
                        out_vec[(i as usize *dimensions as usize)+ (dimensions - j - 1) as usize] = num;
                        out_vec[((dimensions - i - 1) as usize *dimensions as usize)+ (dimensions - j - 1) as usize] = num;
                    }
                }
                out_vec[(dimensions as usize*dimensions as usize)/2] = center;
                normal  = out_vec.iter().sum();
            },
            ConvolutionKernelBuilder::HorizontalPrewitt => {
                let mut temp = vec![0.0 ; dimensions as usize];
                for i in 0..dimensions/ 2 {
                    temp[i as usize] = -1.0;
                    temp[dimensions as usize - i as usize - 1] = 1.0;
                }
                out_vec = temp.iter().cloned().cycle().take(temp.len() * dimensions as usize).collect();
            },
            ConvolutionKernelBuilder::VerticalPrewitt => {
                out_vec = vec![0.0 ; dimensions as usize * dimensions as usize];
                // Bad code, but alright for now
                for i in 0..dimensions/2 {
                    for j in 0..dimensions {
                        out_vec[i as usize * dimensions as usize + j as usize] = 1.0;
                        out_vec[i as usize * dimensions as usize + j as usize + (((dimensions as usize/ 2) + 1) * dimensions as usize )] = -1.0;
                    }
                }
            },
            ConvolutionKernelBuilder::HorizontalSobel => {
                let half = (dimensions / 2) as i32;
                out_vec = vec![0.0 ; dimensions as usize * dimensions as usize];

                for i in 0..dimensions as i32 {
                    for j in 0..dimensions as i32 {
                        out_vec[((i * dimensions as i32) + j) as usize] = (i - half) as f32 / (((i - half)*(i - half)) + ((j - half)*(j - half))) as f32;
                    }
                } 
                out_vec[(dimensions as usize*dimensions as usize)/2] = 0.0_f32;
            },
            ConvolutionKernelBuilder::VerticalSobel => {
                let half = (dimensions / 2) as i32;
                out_vec = vec![0.0 ; dimensions as usize * dimensions as usize];

                for i in 0..dimensions as i32 {
                    for j in 0..dimensions as i32 {
                        out_vec[((i * dimensions as i32) + j) as usize] = (j - half) as f32 / (((i - half)*(i - half)) + ((j - half)*(j - half))) as f32;
                    }
                } 
                out_vec[(dimensions as usize*dimensions as usize)/2] = 0.0_f32;
            },
            ConvolutionKernelBuilder::UnsharpMask => {
                out_vec = vec![-1.0 ; dimensions as usize * dimensions as usize];
                out_vec[(dimensions as usize*dimensions as usize)/2] = dimensions as f32 * dimensions as f32 - 1.0_f32;
            },
        }

        return Kernel(out_vec, dimensions.into(), normal);
    }

}


/// The `Pool` enum contains the pooling methods that can be used in the pipeline.
/// The first field contains dimensionality and the second field stores the stride value.
pub enum Pool {
    Max(u32, u32),
    Min(u32, u32),
    Avg(u32, u32),
}


