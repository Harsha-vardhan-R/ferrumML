//turn the data from a csv into the struct data_frame to modify or train it further.
use std::{error::Error, fs::File, io::BufReader};
use csv::{ReaderBuilder, StringRecord};
use crate::data_frame::{data_frame::*, data_type::DataType};

// todo -- we still need to find a way to load more than the present ram sized datasets, like the panda chunks -- we also need to create completely new data_frame that can dynamically load chunks and a
//new train test splitter and a new accuracy scorer but this is will be taken care of after some time.

//this function may seem dumb for loading the data like 4 times , but i do not think we can allocate the space without that.

pub fn read_csv(file_path : &str , header : bool , category : bool) -> Result<DataFrame, Box<dyn Error>> {
    let start_time = std::time::Instant::now();

    let mut number_of_samples = 0_u32;
    let mut number_of_attributes = 0_u32;

    //opening the file for the first time.
    let file_system = File::open(file_path).unwrap();
    let reader = BufReader::new(file_system);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);
    //first we will count the number of samples 
    for records in csv_reader.into_records() {
        match records {
            Ok(_) => number_of_samples += 1,
            Err(_) => continue,
        };
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
                    data.push(DataType::Floats(vec![0.0_f32 ; number_of_samples as usize]));
                    data_type.push(0);//means float type.
                } else {//if we are not able to parse it.
                    data.push(DataType::Strings(vec![String::new() ; number_of_samples as usize]));//i think we will not get any parsing error because we are not parsing it, i know it sounded dumb but you got what i am saying right?
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
                    data.push(DataType::Category(vec![0_u8 ; number_of_samples as usize]));
                    data_type.push(2);//here it reps category type.
                } else {
                    data.push(DataType::Strings(vec![String::new() ; number_of_samples as usize]));//i think we will not get any parsing error because we are not parsing it, i know it sounded dumb but you got what i am saying right?
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
    let mut count = 0;

    let mut i = 0_usize;//we cannot enumerate because some of the records may fail to parsed, so we will be get up getting a wrong index for it.
    csv_reader.records().for_each(|record| {

        if let Ok(_result) = record {
            for (j , element) in _result.iter().enumerate() {

                match data_type[j] {
                    0 =>  {
                        if let DataType::Floats(data_f32) = &mut data[j]{
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
                        if let DataType::Strings(data_string) = &mut data[j] {
                            //it always parses here, not a problem.
                            data_string[i] = element.to_owned();
                        }
                    },
                    2 =>   {
                        if let DataType::Category(data_vec_u8) = &mut data[j] {
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
            i += 1;
        } else {
            count += 1;
            //eprint!("couldn't parse the record at {}\n", i);
        };
    });

    
    
    let mut headers = vec![String::new() ; number_of_attributes.try_into().unwrap()];
    //setting the headers
    if header {
        let file_system = File::open(file_path)?;
        let reader = BufReader::new(file_system);
        let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

        let temp2 = StringRecord::default();
        let _result = csv_reader.headers().unwrap_or({println!("Parsing headers failed"); &temp2});
        for (i , name) in _result.iter().enumerate() {
            headers[i] = _result.get(i).unwrap_or(" ").to_owned();
        }
    } else {
        for i in 0..number_of_attributes {
            let name = format!( "column{}", i);
            headers[i as usize] = name;
        }
    }

    //creating the complete data frame , then we need not continuously change the size of the vectors when we are reading the files.
    let data_frame_ = DataFrame {
        data,
        headers,
        max_vector,
        min_vector,
        number_of_features: number_of_attributes,
        number_of_samples,
        normalized: false,
    };
    
    println!("parsing failed at {} strings", count);
    println!("Time taken to load the csv as DataFrame into the memory is : {:?}", start_time.elapsed());

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