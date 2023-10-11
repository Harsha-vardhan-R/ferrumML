use std::collections::{HashMap, hash_map::Entry};
use crate::data_frame::{data_type::DataType, return_type::return_type};


///Mainly used when the features represent counts or frequencies of different categories.
/// -for example: like classifying document type, etc...
pub struct MultinomialNb {
    target_classes: Option<DataType>,//we store all the unique target classes , order sensitive. we are going to follow the same order for storing the other parameters.
    target_class_distributions: Vec<usize>,
    total_number_of_cases: usize,
    count_bin: Vec<Vec<HashMap<i32, usize>>>,
    word_count_bin: Vec<i64>,//here we are going to store the total number of words in each class so we need not calculate the probabilities before predicting.
    //taking i64 just in case.
}

///creating the multinomial_NB object.
pub fn multinomial_NB() -> MultinomialNb {
    println!("WARNING! This algorithm assumes that your data represents frequency(assumes the values are integers)");

    MultinomialNb {
        target_classes: None,
        target_class_distributions: vec![],
        total_number_of_cases: 0,
        count_bin: vec![vec![]],
        word_count_bin: vec![]
    }
}

pub trait MLalgo {
    fn fit(&mut self, X_train : &Vec<Vec<f32>>, y_train : &DataType);
    ///uses kernel density estimation rather than just depending on the exact particular probaility.
    fn smooth_fit(&mut self, X_train : &Vec<Vec<f32>>, y_train : &DataType);
}

pub trait predict {
    fn predict(&self, point : &Vec<f32>) -> return_type;
}

//TODO -- the functions reallly have big if else statements which is not good but i am not finding any way to make it better.
impl MLalgo for MultinomialNb {
    
    ///Method to be called on the multinomial_NB struct , will fit the model according to the given data.
    ///assumes the data is the frequency of something occuring so, will be treated as an integer.
    fn fit(&mut self, X_train : &Vec<Vec<f32>>, y_train : &DataType) {

        if let DataType::Category(temp) = y_train {
            
            //creating a hashmap and giving a index to each different category.
            let mut counter: HashMap<u8, usize> = HashMap::new();
            let mut count = 0_usize;
            for i in temp {
                if !counter.contains_key(i) {
                    counter.insert(*i, count);
                    count += 1;
                }
            }

            let mut output_main = vec![vec![HashMap::new()] ; counter.len()];
            let mut distribution_count = vec![0_usize ; counter.len()];
            let mut word_count = vec![0_i64 ; counter.len()];

            //noting down the number of times each feature appeared in each class.
            //and also counting the number of data points in each class and number of total word counts of al features in each class.
            for ( i , row ) in X_train.iter().enumerate() {
                let &index = counter.get(&temp[i]).unwrap();
                distribution_count[index] += 1;
                for (j , &element) in row.iter().enumerate() {
                    word_count[index] += element as i64;//this is to count the probabilities while prediction.
                    //will calculate the sum of frequencies for each class.
                    //entering the values in the hashmap.
                    match output_main[index][j].entry(element as i32) {
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() += 1_usize;
                        },
                        Entry::Vacant(entry) => {
                            entry.insert(1_usize);
                        },
                    }
                }
            }

            let mut in_order_keys: Vec<u8> = vec![0 ; counter.len()];
            for (k , v) in &counter {
                in_order_keys[*v] = *k;
            }

            self.count_bin = output_main;
            self.target_classes = Some(DataType::Category(in_order_keys));
            self.total_number_of_cases = X_train.len().try_into().unwrap();
            self.target_class_distributions = distribution_count.iter().map(|x| *x as usize).collect();
            self.word_count_bin = word_count;

            return;

        } else if let DataType::Strings(temp) = y_train {
                
            //creating a hashmap and giving a index to each different category.
            let mut counter: HashMap<String, usize> = HashMap::new();
            let mut count = 0_usize;
            for i in temp {
                if !counter.contains_key(i) {
                    counter.insert(i.to_string() , count);
                    count += 1;
                }
            }

            let mut output_main = vec![vec![HashMap::new()] ; counter.len()];
            let mut distribution_count = vec![0_usize ; counter.len()];
            let mut word_count = vec![0_i64 ; counter.len()];

            //noting down the number of times each feature appeared in each class.
            //and also counting the number of data points in each class and number of total word counts of al features in each class.
            for ( i , row ) in X_train.iter().enumerate() {
                let &index = counter.get(&temp[i]).unwrap();
                distribution_count[index] += 1;
                for (j , &element) in row.iter().enumerate() {
                    word_count[index] += element as i64;//this is to count the probabilities while prediction.
                    //will calculate the sum of frequencies for each class.
                    //entering the values in the hashmap.
                    match output_main[index][j].entry(element as i32) {
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() += 1_usize;
                        },
                        Entry::Vacant(entry) => {
                            entry.insert(1_usize);
                        },
                    }
                }
            }

            let mut in_order_keys: Vec<String> = vec![String::new() ; counter.len()];
            for (k , v) in &counter {
                in_order_keys[*v] = k.to_string();
            }
            
            self.count_bin = output_main;
            self.target_classes = Some(DataType::Strings(in_order_keys));
            self.total_number_of_cases = X_train.len().try_into().unwrap();
            self.target_class_distributions = distribution_count.iter().map(|x| *x as usize).collect();
            self.word_count_bin = word_count;

            return;

        } else {
            panic!("You cannot train gaussian_NB with float as a target, for this model type");
        }
        
    }

    //we need to implement another kind of fit for which we can use the
    ///using the kernel smoothing technique
    fn smooth_fit(&mut self, X_train : &Vec<Vec<f32>>, y_train : &DataType) {
        
    }

}

//TODO -- way too many type castings, please improve it the code looks messy as shit.

impl predict for MultinomialNb {
    
    fn predict (&self, x : &Vec<f32>) -> return_type {
        
        let mut present_max = (f32::MIN , -1_i32);//-1 to not have any bugs.

        for (i , bin_size) in self.target_class_distributions.iter().enumerate() {
            //initializing with the class priors.
            let mut product_of_conditional = *bin_size as f32 / self.total_number_of_cases as f32;
            for (j , element) in self.count_bin[i].iter().enumerate() {
                product_of_conditional *= match element.get(&(x[j] as i32)) {
                    Some(temp) => (((*temp as f32) * (x[j]))/(self.word_count_bin[i] as f32)).powf(x[j]),
                    None => (1.0 / (self.word_count_bin[i] as f32 + (self.count_bin[0].len() as f32))).powf(x[j]),
                };

                if product_of_conditional > present_max.0 {
                    present_max = (product_of_conditional , i.try_into().unwrap());
                }      
            }
        }

        match self.target_classes.as_ref().unwrap() {
            DataType::Category(temp) => {
                return return_type::Category(temp[present_max.1 as usize]);
            }
            DataType::Strings(temp) => {
                return return_type::Strings(temp[present_max.1 as usize].clone());
            },
            _ => panic!("First train this data then use the predict method, and also you can only train this data on categorical or string targets"),
        }

    }

}