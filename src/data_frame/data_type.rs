

#[derive(Debug, Clone)]
pub enum data_type {
    Strings(Vec<String>),
    Floats(Vec<f32>),
    Category(Vec<u8>),//even if we have only bools or 0 and 1 values a vector of bools will take same amount of space in the memory as of a vector with u8 numbers.
}



//will return the length of the vector wrapped inside of data_type
pub trait length {
    fn len(&self) -> usize ;
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

pub trait print_at_index {
    fn print_at(&self , index : usize) -> ();
}


impl print_at_index for data_type {
    fn print_at(&self , index : usize) -> () {
        match &self {
            data_type::Strings(temp) => {
                print!("{:?}   :  ",temp[index]);
            },
            data_type::Floats(temp) => {
                print!("{:?}   :  ",temp[index]);
            },
            data_type::Category(temp) => {
                print!("{:?}   :  ",temp[index]);
            },
        }
    }
}