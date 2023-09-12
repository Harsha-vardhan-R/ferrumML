use core::panic;
use std::collections::HashMap;
use rayon::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};
use crate::csv_handling::{data_frame, data_type , return_type, print_at_index};

#[derive(Debug)]
pub struct gaussian_NB {
    target_classes: Option<data_type>,//we store all the unique target classes , order sensitive. we are going to follow the same order for storing the other parameters.
    target_class_distributions: Vec<usize>,
    total_number_of_cases: u32,
    means_and_std_devs: Vec<Vec<(f32 , f32)>>,
}

pub fn gaussian_NB() -> gaussian_NB {
    gaussian_NB { 
        target_classes: None, 
        means_and_std_devs: vec![vec![]],
        total_number_of_cases: 0,
        target_class_distributions: vec![],
    }
}

pub trait MLalgo {
    fn fit(&mut self, X_train : &Vec<Vec<f32>> , y_train : &data_type);
}

pub trait predict {
    fn predict(&self, point : &Vec<f32>) -> return_type;
}

impl MLalgo for gaussian_NB {
    
    fn fit(&mut self, X_train : &Vec<Vec<f32>> , y_train : &data_type) {   

        //dbg!(y_train); 

        if let data_type::Category(temp) = y_train {
            
            //creating a hashmap and giving a index to each different category, we may sometimes have the same value for all key value pairs , but it is not worth the risk.
            let mut counter: HashMap<u8, usize> = HashMap::new();
            let mut count = 0_usize;
            for i in temp {
                if !counter.contains_key(i) {
                    counter.insert(*i, count);
                    count += 1;
                }
            }

            //the vector first stores the sum and squares sum for each class for each feature.
            let mut output_main = vec![vec![(0.0_f32 , 0.0_f32) ; X_train[0].len()] ; counter.len()];
            let mut distribution_count = vec![0.0_f32 ; counter.len()];

            for ( i , row ) in X_train.iter().enumerate() {
                let index = counter.get(&temp[i]).unwrap();
                distribution_count[*index] += 1.0_f32;
                for (j , element) in row.iter().enumerate() {
                    output_main[*index][j].0 += *element;
                    output_main[*index][j].1 += element.powf(2.0);
                }
            }

            for i in 0..distribution_count.len() {
                let number = distribution_count[i];
                let number_sqrt = number.sqrt();

                for j in 0..X_train[0].len() {
                    output_main[i][j].0 = if output_main[i][j].0 != 0.0 {
                        output_main[i][j].0 / number
                    } else {
                        1.0_f32 / number//laplacian smoothing.(idk man, it sounds fancy)
                    };
                    //calculating and setting the standard deviation at the same place at which previously the sum of squares are present.
                    output_main[i][j].1 = (output_main[i][j].1 - (distribution_count[i] * output_main[i][j].0.powf(2.0))) / number_sqrt;
                }
            }


            //iterating over a hashmap is the dumbest idea i ever had , it is not fucking folling the order , fuck you chatGPT.
            let mut in_order_keys: Vec<u8> = vec![0 ; counter.len()];
                for (k , v) in &counter {
                    in_order_keys[*v] = *k;
                }

            self.means_and_std_devs = output_main;
            self.target_classes = Some(data_type::Category(in_order_keys));
            self.total_number_of_cases = X_train.len().try_into().unwrap();
            self.target_class_distributions = distribution_count.iter().map(|x| *x as usize).collect();

            return;

        }

        if let data_type::Strings(temp) = y_train {
                //creating a hashmap and giving a index to each different category, we may sometimes have the same value for all key value pairs , but it is not worth the risk.
                let mut counter: HashMap<String, usize> = HashMap::new();
                let mut count = 0_usize;
                for i in temp {
                    if !counter.contains_key(i) {
                        counter.insert(i.to_string(), count);
                        count += 1;
                    }
                }

                //dbg!(&counter);
                //the vector first stores the sum and squares sum for each class for each feature.
                let mut output_main = vec![vec![(0.0_f32 , 0.0_f32) ; X_train[0].len()] ; counter.len()];
                let mut distribution_count = vec![0.0_f32 ; counter.len()];
    
                for ( i , row ) in X_train.iter().enumerate() {
                    let index = counter.get(&temp[i]).unwrap();
                    distribution_count[*index] += 1.0_f32;
                    for (j , element) in row.iter().enumerate() {
                        output_main[*index][j].0 += *element;
                        output_main[*index][j].1 += element.powf(2.0);
                    }
                }

                for i in 0..distribution_count.len() {
                    let number = distribution_count[i];
                    let number_sqrt = number.sqrt();
                    for j in 0..X_train[0].len() {
                        output_main[i][j].0 = if output_main[i][j].0 != 0.0 {
                            output_main[i][j].0 / number
                        } else {
                            1.0_f32 / number//laplacian smoothing.(idk man, it sounds fancy)
                        };
                        //calculating and setting the standard deviation at the same place at which previously the sum of squares are present.
                        output_main[i][j].1 = (output_main[i][j].1 - (distribution_count[i] * (output_main[i][j].0).powf(2.0))) / number_sqrt;
                    }
                }

                let mut in_order_keys: Vec<String> = vec![String::new() ; counter.len()];
                for (k , v) in &counter {
                    in_order_keys[*v] = k.to_owned();
                }
    
                self.means_and_std_devs = output_main;
                self.target_classes = Some(data_type::Strings(in_order_keys));
                self.total_number_of_cases = X_train.len().try_into().unwrap();
                self.target_class_distributions = distribution_count.iter().map(|x| *x as usize).collect();            
            
                return;

        }

        panic!("You cannot train gaussian_NB with float as a target, if it is a !");
    
    }

}

impl predict for gaussian_NB {

    fn predict(&self, point : &Vec<f32>) -> return_type {
        //first we are going to calculate the numerator of the posterior for all the class types.


        let mut present_max = (f32::MIN , -1_i32);//-1 to not have any bugs.

        for i in 0..self.target_class_distributions.len() {
            let mut product_of_conditional = 1.0_f32;
            for j in 0..self.means_and_std_devs[0].len() {
                product_of_conditional *= gaussian_distribution(self.means_and_std_devs[i][j].0 , self.means_and_std_devs[i][j].1, point[j]);
            }
            if (present_max.0 < (product_of_conditional * self.target_class_distributions[i] as f32)) {
                present_max = (product_of_conditional * self.target_class_distributions[i] as f32 , i as i32 );
            }
            //dbg!(&present_max);
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

impl gaussian_NB {
    
    pub fn get_gaussian_vector(&self) {
        let mut counter = 0_usize;
        for class in &self.means_and_std_devs {
            self.target_classes.as_ref().unwrap().print_at(counter);
            for ele in class {
                print!("( {} , {} ) , " , ele.0 , ele.1);
            }
            println!();
        }
        counter += 1;
    }
}

fn gaussian_distribution(mean : f32 , sigma : f32 , x : f32) -> f32 {
    return ((0.3989422_f32/sigma)*(2.71828182_f32.powf(-0.5_f32 * ((x - mean) / sigma).powf(2.0))));
}


