//TODO - confusion matrix.

use crate::{trait_definition::Predict, data_frame::{data_type::{DataType, length}, return_type::ReturnType}};


//model needs to contain the trait predict for this.
pub fn accuracy_score<T : Predict>(model : &T, X_test: &Vec<Vec<f32>> , y_test: &DataType) -> f32 {
    
    let mut correct = 0;

    assert!(X_test.len() == y_test.len() , "The size of the X_test and y_test is not the same");
    
    match y_test {
        DataType::Category(temp) => {
            for (i , point) in X_test.iter().enumerate() {
                if model.predict(point) == ReturnType::Category(temp[i]) {
                    correct += 1;
                }
            }
        },
        //this type is generally not validated through this method but,. just in case.
        DataType::Floats(temp) => {
            for (i , point) in X_test.iter().enumerate() {
                if model.predict(point) == ReturnType::Floats(temp[i]) {
                    correct += 1;
                }
            }
        },
        DataType::Strings(temp) => {
            for (i , point) in X_test.iter().enumerate() {
                if model.predict(point) == ReturnType::Strings(temp[i].clone()) {
                    correct += 1;
                }
                //println!("predicted : {:?} , actual : {:?}" , &model.predict(point) , &return_type::Strings(temp[i].clone()));
            }
        }
    }

    println!("Total test size : {} , guessed correctly : {}" , X_test.len() , correct);

    correct as f32 / X_test.len() as f32

}