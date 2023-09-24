use std::collections::HashMap;
use crate::data_frame::{data_type::data_type, return_type::return_type};


pub struct multinomial_NB {
    target_classes: Option<data_type>,//we store all the unique target classes , order sensitive. we are going to follow the same order for storing the other parameters.
    target_class_distributions: Vec<usize>,
    total_number_of_cases: usize,
    count_bin: Vec<Vec<f32>>,
}

pub fn multinomial_NB() -> multinomial_NB {
    multinomial_NB{
        target_classes: None,
        target_class_distributions: vec![],
        total_number_of_cases: 0,
        count_bin: vec![vec![]],
    }
}

pub trait MLalgo {
    fn fit(&mut self, X_train : &Vec<Vec<f32>>, y_train : &data_type);
}

pub trait predict {
    fn predict(&self, point : &Vec<f32>) -> return_type;
}

impl MLalgo for multinomial_NB {
    
    fn fit(&mut self, X_train : &Vec<Vec<f32>>, y_train : &data_type) {

        if let data_type::Category(temp) = y_train {
            
            //creating a hashmap and giving a index to each different category.
            let mut counter: HashMap<u8, usize> = HashMap::new();
            let mut count = 0_usize;
            for i in temp {
                if !counter.contains_key(i) {
                    counter.insert(*i, count);
                    count += 1;
                }
            }

            let mut output_main = vec![vec![0.0_f32 ; X_train[0].len()] ; counter.len()];
            let mut distribution_count = vec![0_usize ; counter.len()];

            for ( i , row ) in X_train.iter().enumerate() {
                let index = counter.get(&temp[i]).unwrap();
                distribution_count[*index] += 1;
                for (j , &element) in row.iter().enumerate() {
                    output_main[*index][j] += if !(element == 0.0) {
                        element
                    } else {
                        1.0
                    };
                }
            }



            let mut in_order_keys: Vec<u8> = vec![0 ; counter.len()];
            for (k , v) in &counter {
                in_order_keys[*v] = *k;
            }

            self.count_bin = output_main;
            self.target_classes = Some(data_type::Category(in_order_keys));
            self.total_number_of_cases = X_train.len().try_into().unwrap();
            self.target_class_distributions = distribution_count.iter().map(|x| *x as usize).collect();

            return;

        }

        if let data_type::Strings(temp) = y_train {
                
            let mut counter: HashMap<String, usize> = HashMap::new();
            let mut count = 0_usize;
            for i in temp {
                if !counter.contains_key(i) {
                    counter.insert(i.to_owned(), count);
                    count += 1;
                }
            }

            //the vector first stores the sum and squares sum for each class for each feature.
            let mut output_main = vec![vec![0.0_f32 ; X_train[0].len()] ; counter.len()];
            let mut distribution_count = vec![0_usize ; counter.len()];

            for ( i , row ) in X_train.iter().enumerate() {
                let index = counter.get(&temp[i]).unwrap();
                distribution_count[*index] += 1;
                for (j , &element) in row.iter().enumerate() {
                    output_main[*index][j] += element;
                }
            }

            let mut in_order_keys: Vec<String> = vec![String::new() ; counter.len()];
            for (k , v) in &counter {
                in_order_keys[*v] = k.to_string();
            }

            self.count_bin = output_main;
            self.target_classes = Some(data_type::Strings(in_order_keys));
            self.total_number_of_cases = X_train.len().try_into().unwrap();
            self.target_class_distributions = distribution_count.iter().map(|x| *x as usize).collect();

            return;

        }

        panic!("You cannot train gaussian_NB with float as a target, for this model type");
        
    }

}

impl predict for multinomial_NB {
    
    fn predict (&self, x : &Vec<f32> )-> return_type {
        
        let mut present_max = (f32::MIN , -1_i32);//-1 to not have any bugs.

        for i in 0..self.target_class_distributions.len() {
            let mut product_of_conditional = 1.0_f32;
            for (j , element) in self.count_bin[i].iter().enumerate() {
                ////////self.count_bin[i][j]
                /// //need help
            }
        }

        match self.target_classes.as_ref().unwrap() {
            data_type::Category(temp) => {
                return return_type::Category(temp[present_max.1 as usize]);
            }
            data_type::Strings(temp) => {
                return return_type::Strings(temp[present_max.1 as usize].clone());
            },
            _ => panic!("No fucking way this reached here"),
        }
    }

}