//!data_type:
//! The underlying column representation of the data_frame type.
//! can have three different states
//! ---data_type::Float as vec<f32>
//! ---data_type::String as vec<String>
//! ---data_type::Category as vec<u8>
//! inside vectors can me called and mutated but there will be a problem in the uploading of max and min terms 

//a custom iterator which returns an iterator to the mutable reference on the objects .

#[derive(Debug, Clone)]
pub enum DataType {
    Strings(Vec<String>),
    Floats(Vec<f32>),
    Category(Vec<u8>),//even if we have only bools or 0 and 1 values a vector of bools will take same amount of space in the memory as of a vector with u8 numbers.
}



//will return the length of the vector wrapped inside of data_type
pub trait length {
    fn len(&self) -> usize ;
}



impl length for DataType {
    fn len(&self) -> usize {
        match self {
            DataType::Category(temp) => temp.len(),
            DataType::Floats(temp) => temp.len(),
            DataType::Strings(temp) => temp.len(),
        }
    }
}

pub trait print_at_index {
    fn print_at(&self , index : usize) -> ();
}


impl print_at_index for DataType {
    fn print_at(&self , index : usize) -> () {
        match &self {
            DataType::Strings(temp) => {
                print!("{:?}   :  ",temp[index]);
            },
            DataType::Floats(temp) => {
                print!("{:?}   :  ",temp[index]);
            },
            DataType::Category(temp) => {
                print!("{:?}   :  ",temp[index]);
            },
        }
    }
}