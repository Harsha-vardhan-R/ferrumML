//! #Neural networks#
//! 
//! The implementation mainly focuses on flexibility and options.
//! for example : 7 different activation functions, 4 different cost functions, any number of hidden nodes and any number of nodes in each hidden layer, etc...
//! Performance is only dependent on CPU, but future plans include making this code use GPU.
//! Concurrency is present, but can still be improved.
//! Can train continuous or discrete targets, the algo automatically fits the desired type of the output(depending on the type it is given as the target_index in the NeuralNet::new() function).


use std::collections::HashMap;
use std::{f32::consts::E, collections::HashSet};
use fastrand::f32;
use crate::data_frame::data_type::DataType;
use crate::data_frame::return_type::ReturnType;
use crate::data_frame::data_frame;
use crate::{data_frame::data_frame::DataFrame, feature_extraction::tokenisation::special_iterator::SpecialStrDivideall};
use crate::trait_definition::{MLalgo, Predict};


//***************************************
//ACTIVATION FUNCTIONS
//***************************************
///these are the different activation functions,
///the functions and their differentials are defined below.
#[derive(Debug)]
pub enum ActivationFunction {
    ///         x                             ,Identity
    Linear,
    ///         1/(1 + e^-x)                  ,Logistic 
    Sigmoid,
    ///         max(0, x)                     ,Linear rectifier unit
    ReLu,
    ///         (e^x - e^-x)/(e^x + e^-x)     ,Hyperbolic tangent
    Tanh,
    ///         {0 if x < 0; 1 if x >= 0}     ,Binary step 
    BinaryStep,
    ///         ln(1 + e^x)                   ,Soft plus
    SoftPlus,
    ///         {ax if x <= 0; x if x > 0}    ,Leaky ReLU  --  (can have any `a` value)(but default set to 0.01)
    LeakyReLU,
}


//This is the trait implemented for all the functions, for example `ActivationFunction`, etc.....
pub trait functionValueAt {
    fn function_at(&self, x: f32) -> f32;
}

//This is the trait implemented for all the functions that can be differentiated(or approximated), for example `ActivationFunction`, etc.....
pub trait DerivativeValueAt {
    fn derivative_at(&self, x: f32) -> f32;
}


fn max_with_zero(x : f32) -> f32 {
    if x > 0.0 {x} else {0.0}
}
fn tanh(x : f32) -> f32 { 
    let plus = E.powf(x); 
    let minus = E.powf(-x); 
    (plus - minus)/(plus + minus)
}//bad naming cause i want to shoot myself in the foot.

static mut LEAKY : f32 = 0.01_f32;

pub fn set_leaky_value(value : f32) {
    assert!(value > 0.0 && value < 1.0 , "The input value must be between 0.0 and 1.0 exclusively");
    unsafe { LEAKY = value };
}


impl functionValueAt for ActivationFunction {
    
    ///Activation function value at `x`.
    fn function_at(&self, x : f32) -> f32 {
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

}


impl DerivativeValueAt for ActivationFunction {

    ///derivative of the activation function at `x`.
    fn derivative_at(&self, x : f32) -> f32 {
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


//*************************************************
//COST FUNCTIONS
//*************************************************
///different cost functions to be used in Neural Networks.
pub enum CostFunction {
    ///     mean squared error, quadratic cost functiion.
    MSE,
    ///     binary cross entropy, assumes the input values are 0 or 1.
    BCE,
    ///     categorical cross entropy, asssumes target is one-hot encoding. use for multi-class classification.
    CCE,
    ///     mean absolute error, linear cost function.
    MAE,
}


impl CostFunction {
    
    fn cost(&self, present_values : &Vec<f32>, ground_truth : &Vec<f32>) -> f32 {
        match self {
            CostFunction::MSE => {
                let mut to_return = 0.0_f32;
                let mut count = 0.0_f32;
                for (i, float) in present_values.iter().enumerate() {to_return += (float-ground_truth[i]).powf(2.0);count += 1.0;}
                to_return/count
            },
            CostFunction::BCE => {
                let mut to_return = 0.0_f32;
                
                to_return
            },
            CostFunction::CCE => {
                let mut to_return = 0.0_f32;
                
                to_return
            },
            CostFunction::MAE => {
                let mut to_return = 0.0_f32;
                let mut count = 0.0_f32;
                for (i, float) in present_values.iter().enumerate() {to_return += (float - ground_truth[i]).abs();count += 1.0;}
                to_return/count
            },
        }
    }

}



//******************************************
//OUPUT MAP
//******************************************

pub enum OutputMap {
    ///Returns the soft max of the input with the index whose value is at highest.

    SoftMax,
    ///Returns the highest value in the form of vec[highest_value] , and it's index for the given vector.
    ArgMax,
}

impl OutputMap {
    
    fn map(&self, input_vector: &Vec<f32>) -> (Vec<f32>, usize) {
        match self {
            OutputMap::SoftMax => {
                let mut out = vec![0.0_f32; input_vector.len()];
                let mut to_divide = 0.0_f32;
                let mut index_ = 0_usize;
                let mut max_till_now = f32::MIN;

                input_vector.iter().enumerate().for_each(|(index, value)| {
                    out[index] = E.powf(*value);
                    to_divide += out[index];
                    if out[index] > max_till_now {
                        index_ = index;
                        max_till_now = out[index];
                    }
                });

                (out, index_)
            },
            OutputMap::ArgMax => {
                let mut index_ = 0_usize;
                let mut max_till_now = vec![f32::MIN];

                input_vector.iter().enumerate().for_each(|(index, &value)| {
                    if value > max_till_now[0] {
                        index_ = index;
                        max_till_now[0] = value;
                    }
                });

                (max_till_now, index_)
            },
        }
    }

}



//*******************************************************
//*******************************************************
pub struct NeuralNet {
        ///the number of features and the number of classes.
    in_out_size: (usize, usize),
        ///the full structure of the hidden layer, stores the values after the activation functions is used(includes the output vector).
    hidden_layer_active_values: Vec<Vec<f32>>,
        ///number of nodes in each layer (including the input and the output layer).
    pub layer_width: Vec<usize>,//(length of this - 2) gives the number of hidden layers.
        ///weights, a vector that stores the matrices of weights(the conventional indexing way).
        /// 
        ///top vector has a length of (number of hidden layers + 1(output layer))
        /// 
        ///each of the matrices is of the dimensions m*n where m is the size of the next layer while n is the size of the current layer.The weight from a node j to i in the next layer is given by `weight_matrices[hidden_layer_index][i][j]`
        /// 
        ///for example if you have a layer 'i' and a layer 'j', and the layer i is the n'th hidden layer, then the value of the weight from arbitrary i and j is : weight_matrices[n][j][i];(n=0 gives the weights between input and the first hidden layer).
    weight_matrices: Vec<Vec<Vec<f32>>>,
        ///biases, a vector that stores the bias value for each node from the first hidden layer to the output layer(inclusively).
        /// 
        ///length: number of hidden layers + 1;
    bias_vectors: Vec<Vec<f32>>,
        ///the activation function, a.k.a. transfer function.
    pub activation_function: Vec<ActivationFunction>,
        ///Learning step factor
    pub learning_step: f32,
        ///the cost function which we use to train the data using th back propogation.
    pub cost_function: CostFunction,
        ///Stores the type of DataType the target variable is, will be easier for us to select different algos based on the type.
    target_type: crate::data_frame::data_type::DataType,
    target_class: Vec<usize>,
}


impl NeuralNet {
    

    /// Create a NeuralNet object.
    /// 
    /// data_frame : a reference to the data frame object.
    /// 
    /// target_class : the index of the ground truth values in the DataFrame.(only the first element will be considered if your target is a )
    /// 
    /// layer_widths : takes a vector of values which represent the number of neurons in each hidden layer.
    /// 
    /// #for example a `vec![8,8]` creates two hidden layers with 8 neurons each
    /// 
    /// activation_function : takes a Activation function type for example `ActivationFunctions::Sigmoid`
    /// 
    /// learning_step : how fast or slow the values get changed for 
    pub fn new(data_frame: &DataFrame, target_class: Vec<usize>, layer_widths: Vec<usize>, activation_function: Vec<ActivationFunction>, cost_function : CostFunction, learning_step: f32, input_size : usize) -> Self {
        let mut set: HashSet<DataType> = HashSet::new();
        let mut map_n_counter = HashMap::<String, usize>::new();
        let mut count = 0;
        let mut target_type: DataType;


        //the type of data that has to be predicted.
        let output_nodes_here : usize = match &data_frame.data[target_class[0]] {
            ///For training on string types, the number of outputs will be the number of unique tokens is the target column.
            DataType::Strings(temp) => {
                temp.iter().for_each(|x| if (!map_n_counter.contains_key(x)) {map_n_counter.insert(x.to_owned(), count);});
                target_type = DataType::Strings(vec![]);
                map_n_counter.len()
            },
            DataType::Floats(temp) => {
                ///To train on floats we are going to first assert all the indices that are given in the target index vector 
                for each_index in target_class.iter() {
                    match data_frame.data[target_class[*each_index]] {
                        DataType::Floats(_) => {},
                        _ => panic!("All the target indexes should be of the type DataType::Floats, here the index at : {} is not this type" , each_index),
                    }
                }
                target_type = DataType::Floats(vec![]);
                target_class.len()
            },
            DataType::Category(temp) => {
                ///To train on floats we are going to first assert all the indices that are given in the target index vector 
                for each_index in target_class.iter() {
                    match data_frame.data[target_class[*each_index]] {
                        DataType::Category(_) => {},
                        _ => panic!("All the target indexes should be of the type DataType::Category, here the index at : {} is not this type" , each_index),
                    }
                }
                target_type = DataType::Category(vec![]);
                target_class.len()
            },
        };


        //The vector that contains the profile of the current configuration, in the form of [in, h1, h2, ..., out] where in, h.n, out contain the number of nodes in each layer.
        let mut layer_width_ = layer_widths.clone();
        layer_width_.push(output_nodes_here);
        let mut layer_width = vec![input_size];
        layer_width.append(&mut layer_width_);


        ///The weights are randomly given value between 0..1
        let mut weight_matrices: Vec<Vec<Vec<f32>>> = vec![];
        for (present_index, width) in layer_width[1..].iter().enumerate() {
            weight_matrices.push(vec![vec![fastrand::f32(); layer_width[present_index]]; *width]);
        }
        
        ///The biases are set to zero at the start.
        let mut bias_vectors = vec![];
        for width in layer_width[1..].iter() {
            bias_vectors.push(vec![0.0_f32 ; *width]);
        }



        NeuralNet { 
            in_out_size: (input_size , output_nodes_here),
            //the activation values are also initiated to 0.0, not that it matters anyways.
            hidden_layer_active_values: bias_vectors.clone(), 
            layer_width, 
            weight_matrices, 
            bias_vectors,
            activation_function,
            learning_step,
            cost_function,
            target_type,
            target_class, 
        }

    }

    ///Prints the present values of the weights to stdout, as individual matrices.
    pub fn debug_weights(&self) {
        for i in self.weight_matrices.iter() {
            dbg!(i);
        }
    }

    ///prints the bias vectors as they are presently to the stdout.
    pub fn debug_biases(&self) {
        
    } 


    pub fn forward_pass(&self, input_values: &Vec<f32>) {

    }

    pub fn get_layer_detes(&self) -> &Vec<usize> {
        &self.layer_width
    }
 
}



impl MLalgo for NeuralNet {
    fn fit(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType) {
        todo!();
    }
}


impl Predict for NeuralNet {
    fn predict(&self, point : &Vec<f32>) -> ReturnType {
        todo!();
    }
}