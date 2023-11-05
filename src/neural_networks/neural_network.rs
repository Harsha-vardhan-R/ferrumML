//! #Neural networks#
//! 
//! Can have any number of neurons and layers in the hidden layers.
//! 

use std::{f32::consts::E, collections::HashSet};
use ferrumML::data_frame::data_type::DataType;

use crate::{data_frame::data_frame::DataFrame, feature_extraction::tokenisation::special_iterator::SpecialStrDivideall};


//these are the different activation functions,
//the functions and their differentials are defined below.
#[derive(Debug)]
pub enum ActivationFunction {
    Linear,//    x                             Identity
    Sigmoid,//   1/(1 + e^-x)                  Logistic 
    ReLu,//      max(0, x)                     Linear rectifier unit
    Tanh,//      (e^x - e^-x)/(e^x + e^-x)     Hyperbolic tangent
    BinaryStep,//{0 if x < 0; 1 if x >= 0}     Binary step 
    SoftPlus,//  ln(1 + e^x)                   Soft plus
    LeakyReLU,// {ax if x <= 0; x if x > 0}    Leaky ReLU  --  (can have any `a` value)(but default set to 0.01)
}

fn max_with_zero(x : f32) -> f32 { if x > 0.0 {x} else {0.0}}
fn tanh(x : f32) -> f32 { let plus = E.powf(x); let minus = E.powf(-x); (plus - minus)/(plus + minus)}//bad naming cause i want to shoot myself in the foot.

static mut LEAKY : f32 = 0.01_f32;

pub fn set_leaky_value(value : f32) {
    assert!(value > 0.0 && value < 1.0 , "The input value must be between 0.0 and 1.0 exclusively");
    unsafe { LEAKY = value };
}


impl ActivationFunction {
    
    ///Activation function value at `x`.
    pub fn activation_function_at(&self, x : f32) -> f32 {
        match self {
            ActivationFunction::Linear => x,
            ActivationFunction::Sigmoid => 1.0/(1.0+E.powf(-x)),
            ActivationFunction::ReLu => max_with_zero(x),
            ActivationFunction::Tanh => tanh(x),
            ActivationFunction::BinaryStep => {if (x < 0.0) {0.0} else {1.0}},
            ActivationFunction::SoftPlus => (1.0+E.powf(x)).ln(),
            ActivationFunction::LeakyReLU => unsafe {{if (x <= 0.0) {LEAKY*x} else {x}}},
        }
    }

    ///derivative of the activation function at `x`.
    pub fn derivative_at(&self, x : f32) -> f32 {
        match self {
            ActivationFunction::Linear => 1.0,
            ActivationFunction::Sigmoid => {let temp = E.powf(-x); temp*(1.0+temp).powf(-2.0)},
            ActivationFunction::ReLu => if (x > 0.0) {1.0} else {0.0},//f'(0) = 0 (assumption)
            ActivationFunction::Tanh => 1.0-((tanh(x).powf(2.0))),
            ActivationFunction::BinaryStep => 0.0,
            ActivationFunction::SoftPlus => {let temp = E.powf(-x); 1.0/(1.0+temp)},
            ActivationFunction::LeakyReLU => unsafe {if x > 0.0 {1.0} else {LEAKY}},
        }
    }

}


pub struct NeuralNet {
        ///the number of features and the number of classes.
    in_out_size: (usize, usize),
        ///the full structure of the hidden layer, stores the values after the activation functions is used(includes the output vector).
    hidden_layer_active_values: Vec<Vec<f32>>,
        ///number of nodes in each layer (including the input and the output layer).
    layer_width: Vec<usize>,//(length of this - 2) gives the number of hidden layers.
        ///weights, a vector that stores the matrices of weights(the conventional indexing way).
        ///first vector has a length of (number of hidden layers + 1(output layer))
        ///each of the matrices is of the dimensions m*n where m is the size of the next layer while n is the size of the current layer.
        ///for example if you have a layer 'i' and a layer 'j', and the layer i is the n'th hidden layer, then the value of the weight from arbitrary i and j is : weight_matrices[n][j][i];(### n=0 gives the weights between input and the first hidden layer).
    weight_matrices: Vec<Vec<Vec<f32>>>,
        ///biases, a vector that stores the bias value for each node from the first hidden layer to the output layer(inclusively).
        ///length: number of hidden layers + 1;
    bias_vectors: Vec<Vec<f32>>,
        ///the activation function, a.k.a. transfer function.
    pub activation_function: ActivationFunction,
        ///Learning step factor
    pub learning_step: f32,
}


impl NeuralNet {
    
    /// Create a NeuralNet object.
    /// 
    /// data_frame : a reference to the data frame object.
    /// 
    /// target_class : the index of the ground truth values in the DataFrame.
    /// 
    /// layer_widths : takes a vector of vales which represent the number of neurons in each hidden layer.
    /// 
    /// #for example a `vec![8,8]` creates two hidden layers with 8 neurons each
    /// 
    /// activation_function : takes a Activation function type for example `ActivationFunctions::Sigmoid`
    pub fn new(data_frame: &DataFrame, target_class: usize, layer_widths: Vec<usize>, activation_function: ActivationFunction, learning_step : f32) -> Self {
        let mut set: HashSet<DataType> = HashSet::new();
        //the type of data to be predicted.
        match data_frame.data[target_class] {
            crate::data_frame::data_type::DataType::Strings(_) => todo!(),
            crate::data_frame::data_type::DataType::Floats(_) => todo!(),
            crate::data_frame::data_type::DataType::Category(_) => todo!(),
        }

        NeuralNet { 
            in_out_size: (), 
            hidden_layer_active_values: (), 
            layer_width: layer_widths.clone(), 
            weight_matrices: (), 
            bias_vectors: (),
            activation_function,
            learning_step, 
        }
    }

}