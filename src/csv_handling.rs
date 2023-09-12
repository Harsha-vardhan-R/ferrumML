#![allow(non_camel_case_types, non_snake_case, non_camel_case_types, unused_mut, unused_imports)]

use std::{error::Error, fs::File, io::BufReader};
use csv::ReaderBuilder;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, ParallelBridge, IntoParallelRefIterator, IndexedParallelIterator};
use std::collections::HashMap;
use rayon::prelude;
use rand::seq::SliceRandom;
//use std::mem::ManuallyDrop;


#[derive(Debug, Clone)]
pub enum data_type {
    Strings(Vec<String>),
    Floats(Vec<f32>),
    Category(Vec<u8>),//even if we have only bools or 0 and 1 values a vector of bools will take same amount of space in the memory as of a vector with u8 numbers.
}

#[derive(Debug)]
//this is used while returning from a predict function.
pub enum return_type {
    Strings(String),
    Floats(f32),
    Category(u8),
}

//will return the length of the vector wrapped inside of data_type
pub trait length {
    fn len(&self) -> usize ;
}

impl PartialEq for return_type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (return_type::Strings(s1), return_type::Strings(s2)) => s1 == s2,
            (return_type::Floats(f1), return_type::Floats(f2)) => f1 == f2,
            (return_type::Category(c1), return_type::Category(c2)) => c1 == c2,
            _ => false, // Handle other cases or mismatched variants
        }
    }
}

impl length for data_type {
    fn len(&self) -> usize {
        match self {
            data_type::Category(temp) => temp.len(),
            data_type::Floats(temp) => temp.len(),
            data_type::Strings(temp) => temp.len(),
        }
    }
}


pub struct data_frame {
    data: Vec<data_type>,
    headers: Vec<String>,
    number_of_features: u32,
    number_of_samples: u32,
    max_vector: Vec<f32>,//stores the maximum value of each feature.
    min_vector: Vec<f32>,//similarly stores the minimum value.
    normalized: bool,
} 


pub fn get_headers(path : &str , which_features: &Vec<usize> , number_of_features : usize) -> Vec<String> {
    let file_system = File::open(path).unwrap();
    let mut out_vector : Vec<String> = vec![];
    let reader = BufReader::new(file_system);
    let mut match_vector : Vec<usize> = vec![];
    if which_features.is_empty() {//This is to consider only wanted features, if the which features vector is empty that means we want to consider all the features.
        for j in 0..number_of_features {
            match_vector.push(j);
        }
    } else {
        for j in which_features.iter() {
            match_vector.push(*j);
        }
    }
    let mut csv_header = ReaderBuilder::new().has_headers(false).from_reader(reader);
    for header in csv_header.records() {
        let head = header.unwrap();
        for (i , string) in head.iter().enumerate() {
            if match_vector.contains(&i) {
                out_vector.push(string.to_owned());
            }
        }
        break;
    }

    out_vector

}


pub fn read_csv(file_path : &str , header : bool , category : bool) -> Result<data_frame, Box<dyn Error>> {

    let mut number_of_samples = 0_u32;
    let mut number_of_attributes = 0_u32;
    //we are going to fill this using 0 and 1 , if we found that we cannot parse the number , 
    //then we are going to push a 1 or else we are going to push a 0;
    let mut data_type_vector:Vec<u8> = vec![];

    //opening the file for the first time.
    let file_system = File::open(file_path)?;
    let reader = BufReader::new(file_system);
    
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);


    //first we will count the number of samples 
    for records in csv_reader.into_records() {
        let result = records?;
        number_of_samples += 1;
    }

    //opening the file for the second time to calculate the the data_type of the column and count the number of attributes.
    let file_system = File::open(file_path)?;
    let reader = BufReader::new(file_system);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let mut data = vec![]; 
    let mut data_type : Vec<u8> = vec![];   
    //first we will count the number of samples 
    //here we are reading the type of attribute , so if we have any kind of nan or other kind of bullshit here then we are fucked!
    //please make sure not to have any kind of these values be in the 
    if category == false {

        for records in csv_reader.records() {
            let _result = records?;
            for element in _result.iter() {//we need not worry about getting headers because the records method automatically considers from the second row.
                //if we are able to parse the element into a f32.
                if let Ok(_temp) = element.parse::<f32>() {
                    data.push(data_type::Floats(vec![0.0_f32 ; number_of_samples as usize]));
                    data_type.push(0);//means float type.
                } else {//if we are not able to parse it.
                    data.push(data_type::Strings(vec![String::new() ; number_of_samples as usize]));//i think we will not get any parsing error because we are not parsing it, i know it sounded dumb but you got what i am saying right?
                    data_type.push(1);//this means string type.
                }
                number_of_attributes += 1;
            }
            break;//only one loop is required cause data types remain the same , right;).
        }

    } else {

        for records in csv_reader.records() {
            let _result = records?;
            for element in _result.iter() {
                if let Ok(_temp) = element.parse::<u8>() {
                    data.push(data_type::Category(vec![0_u8 ; number_of_samples as usize]));
                    data_type.push(2);//here it reps category type.
                } else {
                    data.push(data_type::Strings(vec![String::new() ; number_of_samples as usize]));//i think we will not get any parsing error because we are not parsing it, i know it sounded dumb but you got what i am saying right?
                    data_type.push(1);
                }
                number_of_attributes += 1;
            }
            break;

        }
    }
    

    //opening for the third time for creating the data set.
    let file_system = File::open(file_path)?;
    let reader = BufReader::new(file_system);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let mut max_vector = return_vector(&data_type, number_of_attributes.try_into().unwrap(), 1);
    let mut min_vector = return_vector(&data_type, number_of_attributes.try_into().unwrap() , 0);

    //the i here give the sample index of the data_set.
    let default_parse_fail = 0_u8;

    let records: Vec<_> = csv_reader.records().collect();
    
    records.iter().enumerate().for_each(|(i, record)| {

        if let Ok(_result) = record {
            for (j , element) in _result.iter().enumerate() {

                match data_type[j] {
                    0 =>  { 
                        if let data_type::Floats(data_f32) = &mut data[j]{
                            //we failed to turn this into a f32 that means it is some nan , null, that kind of shit so we fill it with NAN to identify it easily.
                            let temp = element.parse::<f32>().unwrap_or_else(|_| f32::NAN);
                            data_f32[i] = temp;
                            if temp < min_vector[j] {
                                min_vector[j] = temp;
                            }
                            //i think if we use else if rather than if then we are going to skip all the numbers if the numbers in decending order.
                            if temp > max_vector[j] {
                                max_vector[j] = temp;
                            }
                        }
                    },
                    1 => {
                        if let data_type::Strings(data_string) = &mut data[j] {
                            data_string[i] = element.to_owned();
                        }
                    },
                    2 =>   {
                        if let data_type::Category(data_vec_u8) = &mut data[j] {
                            let temp = element.parse::<u8>().unwrap_or_else(|_| default_parse_fail);//if failed to parse we are going to insert.
                            data_vec_u8[i] = temp;
                            let temp_now = temp as f32;
                            //this may seem odd at first but we need to store the min and max if we are going to normalize this afterwards.
                            if temp_now < min_vector[j] {
                                min_vector[j] = temp_now;
                            }
                            //i think if we use else if rather than if then we are going to skip all the numbers if the numbers in decending order.
                            if temp_now > max_vector[j] {
                                max_vector[j] = temp_now;
                            }
                        }
                    },
                    _ => {}
                }
            }
        };
        //the j will give us the feature attributes.
        
        
        }   
    );
     
    

    let mut headers = vec![String::new() ; number_of_attributes.try_into().unwrap()];
    //setting the headers
    if header {
        let file_system = File::open(file_path)?;
        let reader = BufReader::new(file_system);
        let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

        let _result = csv_reader.headers()?;
        for (i , name) in _result.iter().enumerate() {
            headers[i] = _result.get(i).unwrap().to_owned();
        }
    } else {
        for i in 0..number_of_attributes {
            let name = format!( "column{}", i);
            headers[i as usize] = name;
        }
    }  

    //creating the complete data frame , then we need not continuously change the size of the vectors when we are reading the files.
    let data_frame_ = data_frame {
        data: data,
        headers: headers,
        max_vector: max_vector,
        min_vector: min_vector,
        number_of_features: number_of_attributes,
        number_of_samples: number_of_samples,
        normalized: false,
    };

    Ok(data_frame_)

    }


//creates and returns vectors, boilerplate.
fn return_vector(data_type : &Vec<u8> , size : usize , max_min : u8) -> Vec<f32> {

    let mut new_vec = if max_min == 0 {
        vec![f32::MAX ; size]
    } else {
        vec![f32::MIN ; size]
    };
    
    for i in 0..size {
        if data_type[i] == 1 {
            new_vec[i] = f32::NAN;//if this place is holding a string then we are putting nan value there to avoid further bugs.
        }
    }

    new_vec

}

//functions like normalize and feature and sample point manipulation.
impl data_frame {

    pub fn head(&self) {

        //first we will print the headers
        for heading in &self.headers {
            print!("{:?}", heading);
            print!("      ");
        }

        println!("");


        for i in 0..5 {
            for element in &self.data {
                match element{
                    data_type::Floats(x) => {print!("{}                  ", x[i])},
                    data_type::Strings(y) => {print!("{}                  ", y[i])},
                    data_type::Category(y) => {print!("{}                  ", y[i])},
                }
            }
            println!();
        }
        

    }


    pub fn describe(&self) {

        println!("Number of attributes: {}", self.number_of_features);
        println!("Number of samples: {}", self.number_of_samples);

        let mut column_index = 0;

        let mut count = 0;

        print!("  s.no");
        print!("  column_name");
        print!("            data_type");
        print!(" min");
        println!("     max");

        let width = 22;
        let float_width = 5;
        let number_width = 3;

        for i in &self.data {
            
            match i {
                data_type::Floats(temp) => {
                    //here we are printing the type column name , type , max , min , avg_value ;todo : 25% , 50 % ,75%
                    let mean = 0.0_f32;
                    
                    let column_number = format!("{:number_width$}", column_index + 1);
                    print!("{}->   ", column_number);//the serial numer of the column.
                    let padded_column_name = format!("{:width$}   ", self.headers[column_index]);//the heading of the column.
                    print!("{}", padded_column_name);
                    print!("float   ");                    
                    let mut padded_float = format!("{:<float_width$}   ", self.min_vector[column_index]);
                    print!("{}",padded_float);
                    padded_float = format!("{:<float_width$}   ", self.max_vector[column_index]);
                    println!("{}",padded_float);
                },

                data_type::Strings(temp) => {
                    let mut counter: HashMap<&str, u32> = HashMap::new();

                    for i in temp {
                        let counts = counter.entry(i).or_insert(0);
                        *counts += 1;//incrementing by one each time a value is found.
                    }

                    let column_number = format!("{:number_width$}", column_index + 1);
                    print!("{}->   ", column_number);//the serial numer of the column.//the serial numer of the column.
                    let padded_column_name = format!("{:width$}   ", self.headers[column_index]);//the heading of the column.
                    print!("{}", padded_column_name);//the heading of the column.
                    print!("String   ");           
                    println!("{} unique values" , counter.len());
                    /* if (counter.len() < 25) {
                        for i in counter {
                            println!("                                          {:?}", i);
                        }
                    } */ 
                      
                },

                data_type::Category(temp) => {
                    let mut counter: HashMap<&u8, u32> = HashMap::new();

                    for i in temp.iter() {
                        let counts = counter.entry(i).or_insert(0);
                        *counts += 1;//incrementing by one each time a value is found.
                    }

                    let column_number = format!("{:number_width$}", column_index + 1);
                    print!("{}->   ", column_number);//the serial numer of the column.//the serial numer of the column.
                    let padded_column_name = format!("{:width$}   ", self.headers[column_index]);//the heading of the column.
                    print!("{}", padded_column_name);//the heading of the column.
                    print!("Category   ");           
                    println!("{} unique values" , counter.len());
                    println!("{:?}   ", counter); 
                }
            }
            column_index += 1;
            count += 1;
            if count == 500 {
                break;
            }
        }
    }

    //The get_all is only useful for category and string types.
    pub fn describe_the(&self, column_name : &str , get_all : bool) {
        //getting the index at which the column is located.
        let column_index = self.headers.iter().position(|x| x == column_name).expect("The column name does not exist in the data set");

        let width = 22;
        let float_width = 5;
        let number_width = 3;

        match &self.data[column_index] {
            data_type::Floats(temp) => {
                //here we are printing the type column name , type , max , min , avg_value ;todo : 25% , 50 % ,75%
                let mean = 0.0_f32;
                
                let column_number = format!("{:number_width$}", column_index + 1);
                print!("{}->   ", column_number);//the serial numer of the column.
                let padded_column_name = format!("{:width$}   ", self.headers[column_index]);//the heading of the column.
                print!("{}", padded_column_name);
                print!("float   ");                    
                let mut padded_float = format!("{:<float_width$}   ", self.min_vector[column_index]);
                print!("{}",padded_float);
                padded_float = format!("{:<float_width$}   ", self.max_vector[column_index]);
                
                println!("{}",padded_float);
                for i in 0..5 {
                    print!("{} ,",  temp[i]);
                }
                println!();
            },

            data_type::Strings(temp) => {
                let mut counter: HashMap<&str, u32> = HashMap::new();

                for i in temp {
                    let counts = counter.entry(&i).or_insert(0);
                    *counts += 1;//incrementing by one each time a value is found.
                }

                let column_number = format!("{:number_width$}", column_index + 1);
                print!("{}->   ", column_number);//the serial numer of the column.//the serial numer of the column.
                let padded_column_name = format!("{:width$}   ", self.headers[column_index]);//the heading of the column.
                print!("{}", padded_column_name);//the heading of the column.
                print!("String   ");           
                println!("{} unique values" , counter.len());
                if !get_all {
                    for (i , key) in counter.iter().enumerate() {
                        println!("                                          {:?}", key);
                        if i == 15 {//maximum number of value to be printed, if get all is false.
                            break;
                        }
                    }
                } else {
                    for i in counter {
                        println!("                                          {:?}", i);
                    }
                }
                  
            },

            data_type::Category(temp) => {
                let mut counter: HashMap<&u8, u32> = HashMap::new();

                for i in temp {
                    let counts = counter.entry(&i).or_insert(0);
                    *counts += 1;//incrementing by one each time a value is found.
                }

                println!("{:?}", counter);

                let column_number = format!("{:number_width$}", column_index + 1);
                print!("{}->   ", column_number);//the serial numer of the column.//the serial numer of the column.
                let padded_column_name = format!("{:width$}   ", self.headers[column_index]);//the heading of the column.
                print!("{}", padded_column_name);//the heading of the column.
                print!("Category   ");           
                println!("{} unique values" , counter.len());
                if !get_all {
                    for (i , key) in counter.iter().enumerate() {
                        println!("                                          {:?}", key);
                        if i == 15 {//maximum number of value to be printed, if get all is false.
                            break;
                        }
                    }
                } else {
                    for i in counter {
                        println!("                                          {:?}", i);
                    }
                }
            }
        }

    }



    pub fn rename_columns(&mut self, strings : Vec<&str>) {

        for i in 0..self.headers.len() {
            self.headers[i] = strings[i].to_owned();
        }       
        
    }



    //here we take the name of the name of the column and turn the values into a particular encoding.
    //and also importantly the number of unique values should not exceed 256.
    pub fn encode(&mut self , column_name : &str) {
        //warn!("You can only have upto 256 unique values for this to work or else it is going to throw an error because overflow");
        //getting the index at which the column is located.
        let index = self.headers.iter().position(|x| x == column_name).expect("The column name does not exist in the data set");

        let mut count = 0_u8;//the index value we are going to encode.
        let mut indexer: HashMap<String , u8> = HashMap::new();
        let mut float_indexer: HashMap<f32 , u8> = HashMap::new();

        //giving each unique term an index value, which is basically an encoding.
        match &self.data[index] {
            data_type::Strings(temp) => {
                for i in temp {
                    if !indexer.contains_key(i) {
                        indexer.insert(i.to_owned(), count);
                        count += 1;
                    }
                }
            },
            data_type::Category(temp) => {
                panic!("columns with the type category cannot be encoded");
            },
            data_type::Floats(temp) => {
                panic!("columns with the type float cannot be encoded");
            },
        }

        //allocating the memory 
        let mut new_vector: Vec<u8> = vec![0 ; self.number_of_samples.try_into().unwrap()];

        let temp : &Vec<String> ;

        match &self.data[index] {
            data_type::Strings(temp_) => {
                temp = temp_;
            },
            data_type::Category(_) => panic!("The items in this row are already of the category data_type, no need to encode."),
            data_type::Floats(_) => panic!("You cannot encode float values."),
        }
        for (i , element) in temp.iter().enumerate() {
            new_vector[i] = *indexer.get(element).unwrap();
        }    

        let new_replacer = data_type::Category(new_vector);

        self.data[index] = new_replacer;

        //this is needed so we can normalize this column afterwards if we have to.
        self.max_vector[index] = (indexer.len() - 1) as f32;
        self.min_vector[index] = 0.0_f32;

    }


    pub fn normalize(&mut self) {
        // this is important to normalise the even the input in the predict , because it is still in the 
        //somehow manage to get the max and min values for each of the features from the csv to df cause we are alredy iterating over all the points we need not again iterate and find the max and the min for each feature.
        let number_of_samples_here = self.number_of_samples as usize;
        let number_of_features_here = self.number_of_features as usize;

        let mut min_max = vec![0.0_f32 ; number_of_features_here];

        for i in 0..number_of_features_here {
            if self.min_vector[i] != f32::NAN {
                min_max[i] = self.max_vector[i] - self.min_vector[i];
            } else {
                min_max[i] = f32::NAN;
            }
        }

        //dbg!(&number_of_samples_here);

        //this will store what columns of the category type should be changed.
        let mut to_change: Vec<(usize , &Vec<f32>)> = vec![];

        self.data.par_iter_mut().enumerate().for_each(|(i , column)|
        //here i signifies the column index of the number.
            match column {
                data_type::Floats(temp) => {
                    for j in 0..number_of_samples_here {
                        temp[j] = (temp[j] - self.min_vector[i]) / min_max[i];
                    }
                },
                //cannot normalize types of category and string.
                data_type::Category(temp) => {
                    //here we need to create a new float type column and replace the current one with it.
                    ();///////////////////////////todoooooo
                    /* let mut temp_2 = vec![0.0_f32 ; number_of_samples_here];
                    for j in 0..number_of_samples_here {
                        temp_2[j] = (temp[j] as f32 - self.min_vector[i]) / min_max[i];
                    }
                    to_change.push((i , &temp_2)); */
                },
                data_type::Strings(_) => {
                    ();
                },
            }
        
        
        );


        //setting new min and max, but this will not be trrue if all the values 
        //in the column are same , you need atleast two distinct value for 
        //this to be correct, but assuming.....
        for i in 0..self.min_vector.len() {
            if !self.min_vector[i].is_nan() { 
                self.min_vector[i] = 0.0_f32;
                self.max_vector[i] = 1.0_f32;
            }
        }

        self.normalized = true;  

    } 
 
    /// WARNING - if you want to take out the rows fom 2 to 7 for example, then you need to 
    
    /// remove from the back so that we do not change the index of the next rows and drop ows that we need.
    /// also if this point contains min or max values then we are pretty much fucked up, be careful use this only in cases of emergency.
    /// and also the 0 index here refers to the first row , and not the headers.
    /// pretty inefficient.
    pub fn remove_row(&mut self, index : usize) {
        //removing the value at that row in every column.
        self.data.par_iter_mut().for_each(|i|
            match i {
                data_type::Category(temp) => {
                    temp.remove(index);
                },
                data_type::Floats(temp) => {
                    temp.remove(index);
                },
                data_type::Strings(temp) => {
                    temp.remove(index);
                }
            }
        );
        //updating the number of samples
        self.number_of_samples -= 1;
    } 

    pub fn print_headers(&self) {
        println!("{:?}", self.headers);
    }

    
    pub fn remove_columns(&mut self, which_columns : &Vec<usize>) {
        //need to be really careful cause taking out value at one index in a vector means the index values of all the values after it will shift,
        //so we drop features from the back, which does not change the index values preceeding it.
        let mut which_features_modified = which_columns.clone();
        which_features_modified.sort();
        which_features_modified.reverse();

        //dropping columns in the dataframe.
        for index in which_features_modified.iter() {
            self.data.remove(*index);
        }
        
        
        //dropping the column headers in the self.headers
        for i in &which_features_modified {
            self.headers.remove(*i);
        }

        //changing the number of features.
        self.number_of_features -= which_features_modified.len() as u32; 

        //removing the max and min values of these values.
        for i in &which_features_modified {
            self.max_vector.remove(*i);
            self.min_vector.remove(*i);
        }
        
    } 
    ///'''
    /// data_frame.keep_columns(vector);
    /// '''
    /// This function drops all the columns exept the given columns.
    //internally it just uses the upper funcion
    pub fn keep_columns(&mut self, which_columns : &Vec<u32>) {
        let mut new_feature_set : Vec<usize> = vec![];
        //select all the features you do not want, basically inverting the wanted stuff.
        for i in 0..self.number_of_features {
            if !which_columns.contains(&i) {
                new_feature_set.push(i.try_into().unwrap());
            }
        }
        //here we use the above function to drop the unwanted columns.
        self.remove_columns(&new_feature_set);
    }

    //returns number of rows , number of columns.
    pub fn get_shape(&self) -> (u32, u32) {
        (self.number_of_samples, self.number_of_features)
    }

    
    

    pub fn null_stats(&self) {
        let mut type_of_data = vec![];
        let mut number_of_null = vec![0_u32 ; self.number_of_features.try_into().unwrap()];

        for (i , column) in self.data.iter().enumerate() {
            match column {
                data_type::Category(temp) => {
                    type_of_data.push(2);//you will never get null or nan in this category.
                    
                },
                data_type::Floats(temp) => {
                    type_of_data.push(0);
                    let num_of_null = temp.iter().filter(|x| x.is_nan()).count();
                    number_of_null[i] = num_of_null.try_into().unwrap();
                },
                data_type::Strings(temp) => {
                    type_of_data.push(1);
                    let mut num_of_null = 0_u32;
                    for i in temp.iter() {
                        if i == "null" || i == "NULL" || i == "None" || i == "" {
                            num_of_null+=1;
                        }
                    }
                    number_of_null[i] = num_of_null;
                }
            }
        }

        for (i , type_) in type_of_data.iter().enumerate() {
            print!("{} -> ", i+1);
            if *type_ == 0 {
                println!("{:<20} float   {} null values", self.headers[i] , number_of_null[i]);
            } else if *type_ == 1 {
                println!("{:<20} String   {} null values", self.headers[i] , number_of_null[i]);
            } else {
                println!("{:<20} category   will never have null, automatically replaced with the value 0", self.headers[i]);
            }
        }
    }


}

//interpolation functions
impl data_frame {

    ///interpolates all the missing or nan values.
    /// presently there is only one type , need to implement more types.
    /// dumbfill - fills the empty based on the nearest non nan or node value.
    pub fn interpolate_all(&mut self, method : &str) {
        match method {
            "dumbfill" => self.interpolate_dumbfill(),
            _ => panic!("The given name does not match with any interpolation methods."),
        }
    }

    fn interpolate_dumbfill(&mut self) {
        println!("Warning! If you have nan or none in the first row of your feature then you need to manually change it for this to work.-'dumbfill'");

        self.data.iter_mut().for_each(|column|
            match column {
                data_type::Strings(temp) => {
                    let mut last_non_none = temp[0].clone();
                    for (i , point) in temp.iter_mut().enumerate() {
                        if point == "null" || point == "None" || point == "" || point == "none" {
                            *point = last_non_none.to_string();
                        } else {
                            last_non_none = point.to_string();
                        }
                    }
                },
                data_type::Floats(temp) => {
                    let mut last_non_none = temp[0];
                    for (i , point) in temp.iter_mut().enumerate() {
                        if point.is_nan() {
                            *point = last_non_none;
                        } else {
                            last_non_none = *point;
                        }
                    }
                },
                data_type::Category(temp) => {
                    //this category generally does not have nan or nulls.
                    panic!("program breaking bug found here.");
                },
            }
        );
    }

}

//train test splitter
impl data_frame {
    ///get the index at which the label is located in the data set.
    pub fn get_target_index(&self , target_label : &str) -> Option<usize> {
        for (i , label) in self.headers.iter().enumerate() {
            if target_label == label {
                return Some(i);
            }
        }
        return None;
    }

    ///this method creates a completely new vector which all the ml algos will use so using this function will be always required even for unsupervised or neural network learning.
    //target index is the index you want as the target variable.
    //shuffle -> shuffle randomly shuffles the data points, still no random seed option.
    //after this function , we definetely know that the training is going to be on a vec<vec<f32>> and the target is going to be a data_type.
    pub fn train_test_split(&self , test_size : f32 , target_index : usize , shuffle : bool ) -> (Vec<Vec<f32>> , data_type , Vec<Vec<f32>> , data_type) {

        let test_length = (test_size * self.number_of_samples as f32) as usize;
        let train_length = self.number_of_samples as usize - test_length;

        println!("test_length : {} , train_length : {}" , test_length , train_length);

        let sample_number = self.number_of_samples as usize;
        let feature_number = self.number_of_features as u32 as usize;//we are going to remove the extra target column afterwards.
        let number_of_features_here = feature_number - 1;
        let mut X_train = vec![vec![0.0_f32 ; number_of_features_here] ; train_length];
        let mut X_test = vec![vec![0.0_f32 ; number_of_features_here] ; test_length];
        //creating and shuffling the rows.
        let mut all_rows = vec![0_usize ; sample_number];
        for i in 0..sample_number {
            all_rows[i] = i;
        }
        if shuffle {
            all_rows.shuffle(&mut rand::thread_rng());
        }
        
        //dbg!(&all_rows);
        //selecting only wanted features
        let mut feature_vector = vec![0_usize ; feature_number];
        for i in 0..feature_number {
            feature_vector[i] = i;
        }
        feature_vector.remove(target_index);
        //dbg!(&number_of_features_here, &feature_vector);


        for (enumerated , i) in feature_vector.iter().enumerate() {
            match &self.data[*i] {
                data_type::Category(temp) => {
                    for j in 0..train_length {
                        X_train[j][enumerated] = temp[all_rows[j]] as f32;
                    }
                    for j in train_length..sample_number {
                        X_test[j - train_length][enumerated] = temp[all_rows[j]] as f32;
                    }
                },
                data_type::Floats(temp) => {
                    for j in 0..train_length {
                        X_train[j][enumerated] = temp[all_rows[j]];
                    }
                    for j in train_length..sample_number {
                        X_test[j - train_length][enumerated] = temp[all_rows[j]];
                    }
                },
                data_type::Strings(_) => {
                    panic!("You cannot train with string types , to use this attribute first convert it into a category type");
                }
            }
        }

        if let data_type::Category(temp) = &self.data[target_index] {
            let mut clone = temp.clone();
            for (i , j) in all_rows.iter().enumerate() {
                clone[i] = temp[*j];
            }
            let y_train = data_type::Category(clone[0..train_length].to_vec());
            let y_test = data_type::Category(clone[train_length..sample_number].to_vec());
            return (X_train  , y_train , X_test , y_test);
        }
        
        if let data_type::Floats(temp) = &self.data[target_index] {
            let mut clone = temp.clone();
            for (i , j) in all_rows.iter().enumerate() {
                clone[i] = temp[*j];
            }
            let y_train = data_type::Floats(clone[0..train_length].to_vec());
            let y_test = data_type::Floats(clone[train_length..sample_number].to_vec());
            return (X_train  , y_train , X_test , y_test);
        }

        if let data_type::Strings(temp) = &self.data[target_index] {
            let mut clone = temp.clone();
            for (i , j) in all_rows.iter().enumerate() {
                clone[i] = temp[*j].clone();
            }
            let y_train = data_type::Strings(clone[0..train_length].to_vec());
            let y_test = data_type::Strings(clone[train_length..sample_number].to_vec());
            return (X_train  , y_train , X_test , y_test);
        }
        
        panic!("it did not match with anything????!!!! , this cannot fucking happen.");
        


    } 

}

//Todo.
impl data_frame {
    //replace a value with another value.
    //replace a value witch meets certain conditions with an other value like a formula.
    //creating new data columns by adding values of other two columns.--will be helpful once we implemented the heatmaps for the relation between two heatmaps.
}
