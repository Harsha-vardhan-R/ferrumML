//!Traits which will be used across the files.

use crate::data_frame::data_type::DataType;
use crate::data_frame::return_type::ReturnType;


///all the structs that are for creating and training.
pub trait MLalgo {
    fn fit(&mut self, X_train : &Vec<Vec<f32>> , y_train : &DataType);
}

pub trait Predict {
    fn predict(&self, point : &Vec<f32>) -> ReturnType;
}