//! #Neural networks#
//! 
//! The implementation mainly focuses on flexibility and options.
//! for example : 7 different activation functions, 4 different cost functions, any number of hidden nodes and any number of nodes in each hidden layer and any activation function for each of the layer's nodes, etc...
//! Performance is only dependent on CPU, but future plans include making this code use GPU.
//! Concurrency is present, but can still be improved.
//! Can train continuous or discrete targets, the algo automatically fits the desired type of the output(depending on the type it is given as the target_index in the NeuralNet::new() function).


use std::{collections::{HashMap, HashSet}, f32::consts::E};
use fastrand::f32;
use rand::{random, Rng};
use crate::data_frame::{data_type::DataType, return_type::ReturnType};
use crate::{data_frame::data_frame::DataFrame, trait_definition::MLalgo};


//***************************************
//ACTIVATION FUNCTIONS
//***************************************
///* Different activation functions,
///* The functions and their differentials are defined below.
///* You can create your own ActivationFunction by implementing the traits `FunctionValueAt` and `DerivativeValueAt`
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
    ///         {ax if x <= 0; x if x > 0}    ,Leaky ReLU  --  (can have any `a` value)(but default set to 0.01), can be changed using the set_leaky_value() function.
    LeakyReLU,
}


//This is the first trait that needs to be implemented fif you are creating your own activation function, the second trait is 'DerivativeValueAt'.
pub trait functionValueAt {
    fn function_at(&self, x: f32) -> f32;
}

//This is the second trait that neds to be implemented for a custom activation function, the first one is 'FunctionValueAt'
pub trait DerivativeValueAt {
    fn derivative_at(&self, x: f32) -> f32;
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
            ActivationFunction::Sigmoid => {let temp = E.powf(-x); (1.0 / (1.0 + temp).powf(2.0)) * temp},
            ActivationFunction::ReLu => if (x > 0.0) {1.0} else {0.0},//f'(0) = 0 (assumption)
            ActivationFunction::Tanh => 1.0-((tanh(x).powf(2.0))),
            ActivationFunction::BinaryStep => 0.0,
            ActivationFunction::SoftPlus => {let temp = E.powf(-x); 1.0/(1.0+temp)},
            ActivationFunction::LeakyReLU => unsafe {if x > 0.0 {1.0} else {LEAKY}},
        }
    }

}


///////////////////////////////
///HELPER FUNCTIONS
///////////////////////////////

fn max_with_zero(x : f32) -> f32 {
    if x > 0.0 {x} else {0.0}
}

fn tanh(x : f32) -> f32 { 
    let plus = E.powf(x); 
    let minus = E.powf(-x); 
    (plus - minus)/(plus + minus)
}//bad naming cause i want to shoot myself in the foot.

static mut LEAKY : f32 = 0.01_f32;
///Change the `Leaky` value in `ActivationFunction::LeakyReLU`.
pub fn set_leaky_value(value : f32) {
    assert!(value > 0.0 && value < 1.0 , "The input value must be between 0.0 and 1.0 exclusively");
    unsafe { LEAKY = value };
}



#[derive(Clone)]
//*************************************************
//COST FUNCTIONS
//*************************************************
///different cost functions to be used in Neural Networks.
/// TODO : HUBER LOSS
pub enum CostFunction {
    ///     mean squared error, quadratic cost functiion.
    MSE,
    ///     binary cross entropy, assumes the input values are 0 or 1.(ouput is a single value).
    BCE,
    ///     categorical cross entropy, asssumes more than 2 different category output. use for multi-class classification.
    CCE,
    ///     mean absolute error, linear cost function.
    MAE,
}


impl CostFunction {
    
    fn cost(&self, present_values : &Vec<f32>, ground_truth : &Vec<f32>) -> f32 {
        let count = present_values.len() as f32;
        match self {
            CostFunction::MSE => {
                let mut to_return = 0.0_f32;
                for (i, float) in present_values.iter().enumerate() {to_return += (float-ground_truth[i]).powf(2.0);}
                to_return/count
            },
            CostFunction::BCE => {
                let mut to_return = 0.0_f32;
                for (index, present_node) in present_values.iter().enumerate() {
                    to_return += -(ground_truth[index] * present_node.ln() + (1.0 - ground_truth[index]) * (1.0 - present_node).ln());
                }
                to_return
            },
            CostFunction::CCE => {
                let mut to_return = 0.0_f32;
                for (index, present_node) in present_values.iter().enumerate() {
                    to_return += ground_truth[index] * (*present_node).ln();
                }
                -to_return
            },
            CostFunction::MAE => {
                let mut to_return = 0.0_f32;
                for (i, float) in present_values.iter().enumerate() {to_return += (float - ground_truth[i]).abs();}
                to_return/count
            },
        }
    }

    ///returns a vector of f32's which are the derivatives of the cost w.r.t each respective output node.
    pub fn cost_derivative(&self, present_values : &Vec<f32>, ground_truth : &Vec<f32>) -> Vec<f32> {
        match self {
            CostFunction::MSE => {
                let mut out_vec = vec![0.0_f32; present_values.len()];
                for (value_index, present_value) in present_values.iter().enumerate() {
                    out_vec[value_index] = (2.0*(ground_truth[value_index] - *present_value)/present_values.len() as f32);
                }
                out_vec
            },
            CostFunction::BCE => {
                let mut out_vec = Vec::with_capacity(present_values.len());
                for (index, present_node) in present_values.iter().enumerate() {
                    out_vec.push(present_node - ground_truth[index]);
                }
                out_vec
            },
            CostFunction::CCE => {
                let mut out_vec = Vec::with_capacity(present_values.len());
                for (index, present_node) in present_values.iter().enumerate() {
                    // Derivative of Categorical Cross Entropy with respect to the input.
                    out_vec.push(present_node - ground_truth[index]);
                }
                out_vec
            },
            CostFunction::MAE => {
                let mut out_vec = vec![0.0_f32; present_values.len()];
                for (value_index, present_value) in present_values.iter().enumerate() {
                    // Derivative of Mean Absolute Error with respect to the input
                    if present_value > &ground_truth[value_index] {
                        out_vec[value_index] = 1.0 / present_values.len() as f32;
                    } else {
                        out_vec[value_index] = -1.0 / present_values.len() as f32;
                    }
                }
                out_vec
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
                let index_:usize = 0;
                let max_value = input_vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let softmax_values: Vec<f32> = input_vector.iter().map(|&x| (x - max_value).exp()).collect();


                (softmax_values, 0)
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

///All the types that are possible for the output in NeuralNets.
//{-PRESENTLY NOT BEING USED-}
pub enum OutputCanBe {
    ///For single continuous targets.
    Float(f32),
    ///For multiple continuous targets(Multi-Task Learning).
    FloatVec(Vec<f32>),
    ///For Classification with names.
    String(String),
    ///For categorisation(classification) with numbers.
    Category(u8),
}



//*******************************************************
//*******************************************************
pub struct NeuralNet<T> {
        ///the number of features and the number of classes.
    in_out_size: (usize, usize),
    ///the net values before applying the activation fumction over them.
    net_values: Vec<Vec<f32>>,
        ///the full structure of the hidden layer, stores the values after the activation functions is used(includes the output vector).
    active_values: Vec<Vec<f32>>,
        ///number of nodes in each layer (including the input and the output layer).
    pub layer_width: Vec<usize>,//(length of this - 2) gives the number of hidden layers.
        ///weights, a vector that stores the matrices of weights(the conventional indexing way).
        /// 
        ///top vector has a length of (number of hidden layers + 1(output layer))
        /// 
        ///each of the matrices is of the dimensions m*n where m is the size of the next layer while n is the size of the 
        /// current layer.The weight from a node j to i in the next layer is given by `weight_matrices[hidden_layer_index][i][j]`
        /// 
        ///for example if you have a layer 'i' and a layer 'j', and the layer i is the n'th hidden layer, 
        /// then the value of the weight from arbitrary i and j is : weight_matrices[n][j][i];(n=0 gives the weights between input and the first hidden layer).
    pub weight_matrices: Vec<Vec<Vec<f32>>>,
        ///biases, a vector that stores the bias value for each node from the first hidden layer to the output layer(inclusively).
        /// 
        ///length: number of hidden layers + 1;
    bias_vectors: Vec<Vec<f32>>,
        ///the activation function, a.k.a. transfer function.
        /// You can create your own activation functions just by implementing 
    pub activation_function: Vec<T>,
        ///Learning step factor
    pub learning_step: f32,
        ///the cost function which we use to train the data using back propogation.
    pub cost_function: CostFunction,
        ///Stores the type of DataType the target variable is, will be easier for us to select different algos based on the type.
    target_type: crate::data_frame::data_type::DataType,
        ///Stores the indices of the features we want, typically this will have a size of 1.
    target_indices: Vec<usize>,
        ///The derivatives till now, we are going to store the derivatives till now while coming from the backside.
    chained_derivate: Vec<Vec<f32>>,
        ///This stores the type of stuff to predicted, in-order because the output nodes are going to .
    pub predict_out: OutputCanBe,//vec because we can be in a multi-task situation, where we need to predict multiple outputs in the same network at the same time.(Possible only for the float type)
        ///The output map that is needed to be used to calculate how the output is calculated.
    output_map: OutputMap,
        ///This tells us when to stop the back-propogation.
    pub least_cost : f32,
        ///clipping values for weights and biases, these are set to f32::MAX in the begginning so you do not have any kind of control for growth, use "set_weight_clip_value()" to change this
        ///and this considers the absolute value, so the weights and biases will be clipped based on the magnitudes    
    pub weight_clipping_value : f32,
    pub bias_clipping_value : f32,
        ///number of epoches, can be changed with the method "set_epoch_value()"
    pub epoch_value : usize,
}


/////////////////////////////////////////
/////////////////////////////////////////
impl<T : functionValueAt + DerivativeValueAt> NeuralNet<T> {


    ///* Create a new `NeuralNet` object.
    /// # Arguments
    ///
    ///* data_frame : a reference to the data frame object.
    ///* target_class : the index of the ground truth values in the DataFrame.(only the first element will be considered if your target is a )
    ///* layer_widths : takes a vector of values which represent the number of neurons in each hidden layer.
    /// ```
    /// // For example a `vec![8,8]` creates two hidden layers with 8 neurons each
    /// ```
    ///* activation_function : takes a Activation function type for example `ActivationFunctions::Sigmoid` (NOTE : Any 
    /// type which implements the traits "FunctionValueAt" and "DerivativeValueAt")
    ///* learning_step : how fast or slow the values get updated in the backpropogation, while fitting
    ///* output_map : ignored for float targets, for Categorical and String targets you can choose btween "ArgMax" and "SoftMax" while outputting the values.
    ///* input_size : the number of input features for the present model.(this can be automatically set, will b done in the feature versions).
    pub fn new( data_frame: &DataFrame,
                target_class: Vec<usize>,
                layer_widths: Vec<usize>,
                activation_function: Vec<T>,
                cost_function : CostFunction,
                learning_step: f32,
                output_map : OutputMap,
                input_size : usize ) -> Self {

        assert!(layer_widths.len() != 0, "Cannot initiate with no hidden layers.");

        let mut set: HashSet<DataType> = HashSet::new();
        let mut map_n_counter = HashMap::<String, usize>::new();
        let mut count = 0;
        let mut target_type: DataType;

        assert!(layer_widths.len()+1 == activation_function.len(),
            "You need to provide the activation fumction for all the hidden nodes and also the ouput node, 
            if you do not want give any activation function, just give itActivationFunction::Linear");
        
        let output_nodes_here : usize;
        
        match &data_frame.data[target_class[0]] {
            ///For training on string types, the number of outputs will be the number of unique tokens is the target column.
            DataType::Strings(temp) => {
                temp.iter().for_each(|x| if (!map_n_counter.contains_key(x)) {map_n_counter.insert(x.to_owned(), count);});
                target_type = DataType::Strings(vec![]);
                map_n_counter.len();
                output_nodes_here = 1;
            },
            DataType::Floats(temp) => {
                ///To train on floats we are going to first assert all the indices that are given in the target index vector 
                for each_index in target_class.iter() {
                    match data_frame.data[*each_index] {
                        DataType::Floats(_) => {},
                        _ => panic!("All the target indexes should be of the type DataType::Floats, here the index at : {} is not this type" , each_index),
                    }
                }
                //this will not be used anywhere, the Category and Float counterparts will have a major use though.
                target_type = DataType::Floats(vec![]);
                output_nodes_here = target_class.len();
            },
            DataType::Category(temp) => {
                ///To train on category 
                for each_index in target_class.iter() {
                    match data_frame.data[target_class[*each_index]] {
                        DataType::Category(_) => {},
                        _ => panic!("All the target indexes should be of the type DataType::Category, here the index at : {} is not this type" , each_index),
                    }
                }
                target_type = DataType::Category(vec![]);
                output_nodes_here = 1;
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
            weight_matrices.push(vec![vec![0.0_f32; layer_width[present_index]]; *width]);
        }
        Self::fill_rand(&mut weight_matrices);
        
        ///The biases are set to zero at the start.
        let mut bias_vectors = vec![];
        for width in layer_width[1..].iter() {
            bias_vectors.push(vec![0.0_f32 ; *width]);
        }
        Self::fill_rand_2(&mut bias_vectors);

        eprintln!("WARNING: The default values are set for the following fields please use the 'set_<field_name>()' methods to change the respective fields
        least_cost = 0.01
        bias_clipping_value = NO CLIP
        weights_clipping_value = NO CLIP
        epoches = 50");

        NeuralNet {
            in_out_size: (input_size , output_nodes_here),
            chained_derivate: bias_vectors.clone(),
            net_values: bias_vectors.clone(),
            active_values: bias_vectors.clone(),//the activation values are also initiated to 0.0, not that it matters anyways.
            layer_width, 
            weight_matrices, 
            bias_vectors,
            activation_function,
            output_map,
            learning_step,
            cost_function,
            target_type,
            target_indices: target_class, 
            predict_out : OutputCanBe::Float(0.0),
            least_cost : 0.01,
            weight_clipping_value : f32::MAX,
            bias_clipping_value : f32::MAX,
            epoch_value : 50,
        }

    }

    ///this will initialize the weights based on "Xavier/Glorat" Initialization for all the layers.
    /// should be called before the fitting process.
    /// works better with tanh and sigmoid activation functions.
    pub fn xavier_weights(&mut self) {
        //we can directly use this 
        for (input_index, input_node_len) in self.layer_width.iter().skip(1).enumerate() {
            for weight_node_group in self.weight_matrices[input_index].iter_mut() {
                for node_prev_weight in weight_node_group.iter_mut() {
                    *node_prev_weight = rand::thread_rng().gen_range((-1.0/(*input_node_len as f32).sqrt())..((1.0)/(*input_node_len as f32).sqrt()));
                }
            }
        }
    }

    ///this will initialize the weights based on "He" Initialization for all the layers.
    /// should be called before the fitting process.
    /// works better with ReLU and other activation functions.
    pub fn he_weights(&mut self) {
        for (input_index, input_node_len) in self.layer_width.iter().skip(1).enumerate() {
            for weight_node_group in self.weight_matrices[input_index].iter_mut() {
                for node_prev_weight in weight_node_group.iter_mut() {
                    *node_prev_weight = rand::thread_rng().gen_range((-2.0 /(*input_node_len as f32).sqrt())..((2.0) / (*input_node_len as f32).sqrt()));
                }
            }
        }
    }


    pub fn change_minimum_cost(&mut self, minimum_cost: f32) {
        if (minimum_cost.is_sign_negative()) {panic!("Expected a positive value here");}
        self.least_cost = minimum_cost;        
    }

    fn fill_rand(input : &mut Vec<Vec<Vec<f32>>>) {
        input.iter_mut().for_each(|x| {
            x.iter_mut().for_each(|y| {
                y.iter_mut().for_each(|z| {
                    *z = fastrand::f32();
                });
            });
        });
    }

    fn fill_rand_2(input : &mut Vec<Vec<f32>>) {
        input.iter_mut().for_each(|x| {
            x.iter_mut().for_each(|y| {
                *y = random();
            });
        });
    }

    fn fill_same(input : &mut Vec<Vec<Vec<f32>>>, same : f32) {
        input.iter_mut().for_each(|x| {
            x.iter_mut().for_each(|y| {
                y.iter_mut().for_each(|z| {
                    *z = same;
                });
            });
        });
    }

    //+++++++++++++++++++++++++++++
    //DEBUG FUNCTIONS//////////////
    //+++++++++++++++++++++++++++++

    ///Prints the present values of the weights to stdout, as individual matrices.
    pub fn debug_weights(&self) {
        dbg!(&self.weight_matrices);
    }

    ///prints the bias vectors as they are presently to the stdout.
    pub fn debug_biases(&self) {
        dbg!(&self.bias_vectors);
    }

    ///prints the current activation values of the stdout.(it does not contain the input values but does contain the last output layer values).
    pub fn debug_activation_values(&self) {
        dbg!(&self.active_values);
    }

    //prints the current net values of all the layers including the last layer to the stdout
    pub fn debug_net_values(&self) {
        dbg!(&self.net_values);
    }

    pub fn debug_chained_derivaives(&self) {
        dbg!(&self.chained_derivate);
    }


    ////////////////////////////////////
    /// SET FUNCTIONS///////////////////
    ////////////////////////////////////
    
    ///Returns the number of nodes in each layer including the input and the ouput layers.
    pub fn get_layer_detes(&self) -> &Vec<usize> {
        &self.layer_width
    }

    ///setting the weight clipping value
    pub fn set_weight_clip_value(&mut self, value : f32) {
        self.weight_clipping_value = f32::abs(value);
    }

    ///setting the bias clipping value
    pub fn set_bias_clip_value(&mut self, value : f32) {
        self.bias_clipping_value = f32::abs(value);
    }

    ///setting the number of epoches
    pub fn set_epoch_value(&mut self, value : usize) {
        self.epoch_value = value;
    }


    /////////////////////////////////////////////
    ///The last layer of activation gives the current active values, after this function is run once.
    /// 
    ///Returns the last(output layer), used in the prediction function.
    pub fn feed_forward(&mut self, input_values: &Vec<f32>) -> &Vec<f32> {

        assert!(input_values.len() == self.layer_width[0], "The input dimensionality must be same for both the NeuralNet and the present input_values");

        let mut present: f32;
        //setting the first layer ourselves(outside the loop)
        //for each node in the first layer
        for (node_index, first_layer_node) in self.net_values[0].iter_mut().enumerate() {
            present = 0.0;//initializing for this iteration
            for (input_index, input_value) in input_values.iter().enumerate() {
                present += *input_value*self.weight_matrices[0][node_index][input_index];
            }
            //layer node value is the sum of net product with weights and bias at this node.
            *first_layer_node = present + self.bias_vectors[0][node_index];
            self.active_values[0][node_index] = self.activation_function[0].function_at(*first_layer_node);
        }
        //for the remaining layers.
        for (layer_index, layer) in self.net_values.iter_mut().skip(1).enumerate() {//for each layer(from the second layer)
            let true_layer_index = layer_index + 1;
            for (node_index, node_value) in layer.iter_mut().enumerate() {
                present = 0.0;
                for (input_for_this_index, input_for_this_value) in self.active_values[layer_index].iter().enumerate() {
                    present += *input_for_this_value*self.weight_matrices[true_layer_index][node_index][input_for_this_index];
                }
                *node_value = present + self.bias_vectors[true_layer_index][node_index];
                self.active_values[true_layer_index][node_index] = self.activation_function[true_layer_index].function_at(*node_value);
            }
        }

        &self.net_values[self.active_values.len()-1]

    }


    //The name is self-explanatory.
    //Takes in the value of the output and what `should` they be.
    //Modifies the values of weights and the biases, to make the 'cost' less.
    // This function does NOT use any kind of parallelism.
    pub fn feed_forward_back_propogate(&mut self, input_values: &Vec<f32>, ground: &Vec<f32>) -> f32 {
        //feeding forward
        self.feed_forward(input_values);
        //the `chained derivative` field of the struct stores the chain of derivaives till the 'net' value of the node at the respective index in the other fields such as active values and net values.

        ///BACK PROPOGATION. 
        //Setting the chained derivative values for the last layer.
        let total_cost = self.cost_function.cost(&self.active_values[self.active_values.len()-1], ground);
        let out_nodes_cost_derivative = self.cost_function.cost_derivative(&self.net_values[self.net_values.len()-1], ground);
        //dbg!(total_cost, &out_nodes_cost_derivative);
        let last_index = self.active_values.len()-1;
        //for each node in the ouput layer.
        for (node_index, out_node) in out_nodes_cost_derivative.iter().enumerate() {
            // self.chained_derivate[last_index][node_index] = *out_node*self.activation_function[last_index].derivative_at(self.net_values[last_index][node_index]);
            self.chained_derivate[last_index][node_index] = *out_node*self.activation_function[last_index].derivative_at(self.active_values[last_index][node_index]);
            //updating the weights(for the last layer)
            //for each weight to the present node.
            for (prev_layer_index, weight_value) in self.weight_matrices[last_index][node_index].iter_mut().enumerate() {
                *weight_value -= self.learning_step*(self.chained_derivate[last_index][node_index]*(self.active_values[last_index-1][prev_layer_index]));
                if f32::abs(*weight_value) > f32::abs(self.weight_clipping_value) {
                    //we gotta preserve the sign though.
                    if *weight_value < 0.0 {
                        *weight_value *= -self.weight_clipping_value;
                    } else {
                        *weight_value = self.weight_clipping_value;
                    }
                }
            }
            //updating the biases(for the last layer)
            self.bias_vectors[last_index][node_index] -= self.learning_step*(self.chained_derivate[last_index][node_index]);
            if f32::abs(self.bias_vectors[last_index][node_index]) > f32::abs(self.bias_clipping_value) {
                if self.bias_vectors[last_index][node_index] < 0.0 {
                    self.bias_vectors[last_index][node_index] *= -self.bias_clipping_value;
                } else {
                    self.bias_vectors[last_index][node_index] *= self.bias_clipping_value;
                }
            }
        }

        let mut CurrentSumOfProduct: f32;
        /// FOR ALL THE REMAINING LAYERS TILL THE FIRST LAYER.
        /// of your neural network only has one hidden layer, this loop will not even run.
        // for each layer, from the last second layer.
        for layer_index in (0..last_index).rev() {
            //for each layer we are going to calculate the loss of each node first,
            //then we are going to modify the layer's weights and biases related to that layer.
            //because we are storing the chained values of the derivatives till then we need not calculate all those values again(I mean obviously we are not going to calculate the full value again . it is dumb AF)
            for (node_index, node_in_layer) in self.active_values[layer_index].iter().enumerate() {
                //in this case, the loss is not just directly derived, because each node effects all the nodes that are in the next layer.
                //the derivative is going to be the sum of all the products of the weights from the present node to the next layer and the node values of the next layer.
                //so we need to calculate it while giving determining the value of the chained derivative.
                CurrentSumOfProduct = 0.0_f32;
                for (next_index, next_value) in self.chained_derivate[layer_index+1].iter().enumerate() {
                    CurrentSumOfProduct += *next_value*self.weight_matrices[layer_index+1][next_index][node_index];
                }
                //update the chained derivative value for this node in this iteration(for the sample point).
                self.chained_derivate[layer_index][node_index] = CurrentSumOfProduct*self.activation_function[layer_index].derivative_at(self.active_values[layer_index][node_index]);
                
                
                if (layer_index != 0) {
                    //updating the related weights and the related bias.
                    for (weight_index, weight_value) in self.weight_matrices[layer_index][node_index].iter_mut().enumerate() {
                        *weight_value -= self.learning_step*(self.chained_derivate[layer_index][node_index]*self.active_values[layer_index-1][weight_index]);
                        if f32::abs(*weight_value) > f32::abs(self.weight_clipping_value) {
                            if *weight_value < 0.0 {
                                *weight_value *= -self.weight_clipping_value;
                            } else {
                                *weight_value = self.weight_clipping_value;
                            }
                        }
                    }
                    self.bias_vectors[layer_index][node_index] += self.learning_step*(self.chained_derivate[layer_index][node_index]);
                    if f32::abs(self.bias_vectors[layer_index][node_index]) > f32::abs(self.bias_clipping_value) {
                        if self.bias_vectors[layer_index][node_index] < 0.0 {
                            self.bias_vectors[layer_index][node_index] *= -self.bias_clipping_value;
                        } else {
                            self.bias_vectors[layer_index][node_index] *= self.bias_clipping_value;
                        }
                    }
                }
            }
        }

        //FOR THE WEIGHTS FROM THE INPUT TO THE FIRST HIDDEN LAYER.
        for (node_index, weight_values) in self.weight_matrices[0].iter_mut().enumerate() {
            for (input_index, weight_value) in weight_values.iter_mut().enumerate() {
                *weight_value -= self.learning_step*(input_values[input_index]*self.chained_derivate[0][node_index]);
                if f32::abs(*weight_value) > f32::abs(self.weight_clipping_value) {
                    if *weight_value < 0.0 {
                        *weight_value *= -self.weight_clipping_value;
                    } else {
                        *weight_value = self.weight_clipping_value;
                    }
                }
            }
            self.bias_vectors[0][node_index] -= self.learning_step*(self.chained_derivate[0][node_index]);
            if f32::abs(self.bias_vectors[0][node_index]) > f32::abs(self.bias_clipping_value) {
                if self.bias_vectors[0][node_index] < 0.0 {
                    self.bias_vectors[0][node_index] *= -self.bias_clipping_value;
                } else {
                    self.bias_vectors[0][node_index] *= self.bias_clipping_value;
                }
            }
        }

        return total_cost;
        
    }

    //Basically curve fitting.
    ///* Curve fitting on a single continuous output.
    fn fit_float(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType) {
        //The basic back-prop when to to stop loop.

        let ground: &Vec<f32> = match y_train {
            DataType::Floats(temp) => temp,
            _ => panic!("Wrong type!"),
        };
        
        let mut present_cost: f32;
        let mut placeholder_vector = vec![0.0_f32];
        //for each epoch in the total number of epoch values.
        for epoch_index in 0..self.epoch_value {
            let mut present_cost_max = f32::MIN;
            // for each data point, backpropogate and update the weights.
            for (index, present_theta) in X_train.iter().enumerate() {
                //setting the ground truth value for this sample.
                placeholder_vector[0] = ground[index];
                //this function first feeds forward, then back-propogates.
                present_cost = self.feed_forward_back_propogate(present_theta, &placeholder_vector);
                //updating the present cost if it is the biggest till now in the present epoch.
                if (present_cost > present_cost_max) {
                    present_cost_max = present_cost;
                }
            }

            println!("Epoch: [{}/{}], Maximum cost: {}", epoch_index+1, self.epoch_value, present_cost_max);
            
            if (present_cost_max < self.least_cost) {
                println!("The least cost value of {} is reached in just {} epoches", self.least_cost, epoch_index+1);
                break;
            }
        }

    }
   
   

    fn fit_string(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType) {
        todo!();
    }

    /// This is called when the target is of the type 'DataType::Category'
    /// use the cost functions : "CostFunction::BCE" and "CostFunction::CCE" for categorical targets.
    /// there will be a warning if tried to train with BCE but there are more than 
    fn fit_category(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType) {
        //The basic back-prop when to to stop loop.

        let ground: &Vec<u8> = match y_train {
            DataType::Category(temp) => temp,
            _ => panic!("Wrong type!"),
        };

        let mut present_cost: f32;
        let mut placeholder_vector = vec![0.0_f32];
        //for each epoch in the total number of epoch values.
        for epoch_index in 0..self.epoch_value {
            let mut present_cost_max = f32::MIN;
            // for each data point, backpropogate and update the weights.
            for (index, present_theta) in X_train.iter().enumerate() {
                //setting the ground truth value for this sample.
                placeholder_vector[0] = ground[index] as f32;
                //this function first feeds forward, then back-propogates.
                present_cost = self.feed_forward_back_propogate(present_theta, &placeholder_vector);
                //updating the present cost if it is the biggest till now in the present epoch.
                if (present_cost > present_cost_max) {
                    present_cost_max = present_cost;
                }
            }

            println!("Epoch: [{}/{}], Maximum cost: {}", epoch_index+1, self.epoch_value, present_cost_max);
            
            if (present_cost_max < self.least_cost) {
                println!("The least cost value of {} is reached in just {} epoches", self.least_cost, epoch_index+1);
                break;
            }
        }

    }
    
    

    //Multiple curve fitting.
    pub fn fit_multi_task_float(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType) {
        todo!();
    }

    fn predict_string() {
        todo!();
    }

    ///Returns the values of last layer(output layer),
    ///* Here only one input is being expected and also returned from the feed_forward() method,
    /// so we do not need to worry about
    pub fn predict_float(&mut self, input_values: &Vec<f32>) -> f32 {
        return self.feed_forward(input_values)[0];
    }

    fn predict_category() {
        todo!();
    }

    fn predict_multi_task_float() {
        todo!();
    }

    fn predict(&self, point : &Vec<f32>) -> ReturnType {
        todo!();
    }

}


impl<T : functionValueAt + DerivativeValueAt> MLalgo for NeuralNet<T> {
    ///The fit function automatically changes the type of algorithm used based on the target type.
    fn fit(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType) {
        let start_time = std::time::Instant::now();
        match &self.target_type {
            DataType::Strings(_) => self.fit_string(X_train, y_train),
            DataType::Floats(_) => if self.target_indices.len() == 1 {
                self.fit_float(X_train, y_train);
            } else {
                panic!("Please use 'ObjectName.fit_multi_task_float(X_train, y_train)' for this purpose");
            },
            DataType::Category(_) => self.fit_category(X_train, y_train),
        }
        eprintln!("Time required to train this : {:?}", start_time.elapsed())
    }
}


