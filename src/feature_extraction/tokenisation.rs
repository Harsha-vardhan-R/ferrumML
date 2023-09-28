//!FEATURE EXTRACTION:
//!Extract the wanted attributes from the data.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use rand::seq::index;

use crate::data_frame::data_type::data_type;
use crate::data_frame::data_frame::data_frame;

pub struct Tokens<'a> {
    column_index: HashMap<&'a str, usize>,//used to store the index at which each token is present.
    token_vector: Vec<Vec<usize>>,//the sparse(typically) matrix which we are going to store of the repetation.
    token_distribution: Vec<usize>,//store the rarity of the each word in a row matrix.
}

pub struct SpecialStr<'a> {
    string: &'a str,
    back: usize,//index of the back of the &str substring.
}

impl<'a> SpecialStr<'a> {
    pub fn new(input : &'a str) -> Self {
        SpecialStr{
            string : input,
            back: 0,
        }
    }
}

//to split at all the places which are not 
fn is_special(c: char) -> bool {
    !c.is_ascii_alphanumeric()
}

impl<'a> Iterator for SpecialStr<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let input_string: &str = self.string;
        let max_index = self.string.len();

        for i in self.back..max_index {
            if !is_special(self.string.chars().nth(i).unwrap()) {
                for j in (i+1)..max_index {
                    if is_special(self.string.chars().nth(j).unwrap()) || j == max_index {
                        self.back = j;
                        return Some(&input_string[i..j]);
                    }
                }
            }
        }

        None
    }
}


impl<'a> Tokens<'a> {

    pub fn new() -> Tokens<'a> {
        Tokens {
            column_index: HashMap::new(),
            token_vector: vec![vec![]], 
            token_distribution: vec![] 
        }
    }

    //possible only for the string data type.
    pub fn tokenise(&mut self, frame : &data_frame, column_index : usize) {

        //not preallocating the memory because we do not know the number of individual words we are going to come across.
        let mut token_distribution: Vec<usize> = vec![];
        let mut count: usize = 0_usize;
        let mut sparse: Vec<Vec<usize>> = vec![vec![]];
        let mut mapper: HashMap<&str, usize> = HashMap::new();

      /*   match &frame.data[column_index] {
            data_type::Strings(temp) => {
                //for each sentence in the given column.
                for sentence in temp {
                    //comparing each word after making it lowercase.
                    for word in sentence.split_whitespace().into_iter() {
                        //if the word is already occupied then we are going to just add 1 to the token distribution at that index
                        //or we are going to insert this value with the value 1. 
                        match  self.column_index.entry(word) {
                            Entry::Occupied(temp) => {
                                let &index_of = temp.get();//where do we need to locate the 
                                token_distribution[index_of] += 1;
                                //sparse[index_of][]
                            },
                            Entry::Vacant(entry) => {
                                
                            },
                        }
                    }
                }
            },
            _ => panic!("You cannot tokenise the float or the category data type"),
        } */
    }


}