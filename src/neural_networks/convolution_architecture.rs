use super::{neural_network::{functionValueAt, DerivativeValueAt}, convolution_kernel::Pool};




/// Struct that actually stores the details about the user defined architecture.
pub struct ConvolutionArchitecture<U : functionValueAt> {
    architecture : Vec<((u32, u32, u32), U, Option<Pool>)>,
    feature_maps : Vec<Vec<Vec<f32>>>,
    /// kernel value for each of the layer.
    weight_kernels : Vec<Vec<Vec<Vec<f32>>>>,
    // active values
    active_values : Vec<Vec<Vec<Vec<f32>>>>,
    // bias values for each feature map.
    bias_values : Vec<Vec<Vec<Vec<f32>>>>,
    /// layer dimensions including the first and the last layer. 
    layer_dimensions : Vec<(u32, u32, u32)>,
    workgroup_size : (u32, u32, u32),
    batch_size : u32,
}




impl<U : functionValueAt + DerivativeValueAt> ConvolutionArchitecture<U> {

    /// Creating a new convolution architecture,
    /// mostly to put before a pipe that contains convolution layers in the front an feed-forward network.
    /// 
    /// arguments :
    /// - input size : the first two dimensions are the dimensions of the input image, the third dimension is the number of channels in the present image.
    ///    the fourth field is for the input pooling.
    /// - hidden_convolutional_layers : type -> `Vec<((u32, u32, u32), U, Option<Pool>, u8)>`,
    ///     for each layer need to mention : 
    /// 
    /// * (u32, u32, u32) -> size of the kernel for this layer.
    /// * `U : functionValueAt + DerivativeValueAt` an activation function 
    /// * Option<Pool> is for pooling, selecting the option `none` will not pool the output.
    /// * and the last u8 argument is for the padding amount, set to 0 for no padding.
    fn new( input_size : (u32, u32, u32, u8),
            hidden_convolutional_layers : Vec<((u32, u32, u32), U, Option<Pool>, u8)>,
            batch_Size : u32) -> Self {
        
        let mut layer_dimensions: Vec<(u32, u32, u32, u8)> = vec![input_size];
        
        // first we are going to verify whether this kind of architecture can exist or not.
        for (index, layer) in hidden_convolutional_layers.iter().enumerate() {
            let previous_layer_dimen = layer_dimensions[index];
            match &layer.2 {
                Some(temp) => {
                    
                },
                None => {

                },
            }
        }




        ConvolutionArchitecture {
            architecture: todo!(),
            feature_maps: todo!(),
            weight_kernels: todo!(),
            active_values: todo!(),
            bias_values: todo!(),
            layer_dimensions : todo!(),
            workgroup_size: todo!(),
            batch_size: batch_Size,
        }
    }
}