//!FEATURE EXTRACTION:
//!Extract the wanted attributes from the data.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use crate::data_frame::data_type::data_type;
use crate::data_frame::data_frame::data_frame;

pub struct Tokens {
    column_index: HashMap<String, usize>,//used to store the index at which each token is present.
    token_vector: Vec<Vec<usize>>,//the sparse(typically) matrix which we are going to store of the repetation.
    token_distribution: Vec<usize>,//store the rarity of the each word in a row matrix.
}


//special whitespace and special character dividing iterator.
pub struct SpecialStr<'a> {
    string: &'a str,
    back: usize,//index of the back of the &str substring.
}

impl<'a> SpecialStr<'a> {
    pub fn new(input : &'a str) -> Self {
        SpecialStr {string: input, back: 0}
    }
}

//to split at all the places which are special.
//but we need to give some special importance to the '?', '!', '', ''

//anything which is not a alphanumeric or a whitespace.
fn is_special(c: char) -> bool {
    !c.is_ascii_alphanumeric() && !c.is_whitespace()
}

//this iterator divides into "I am an assHoLe!." into an iterator which gives out ("i" , "am", "an" , "asshole" , "!", ".")
//we are going to split at the whitespaces and any special characters.

impl<'a> Iterator for SpecialStr<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let input_string: &str = self.string;
        let max_index = self.string.len();

        for front in self.back..max_index {
            //if the present char is a special character just return it by itself.
            if is_special(self.string.chars().nth(front).unwrap()) {
                self.back += 1;
                return Some(&input_string[self.back-1..self.back]);
            } else if !self.string.chars().nth(front).unwrap().is_whitespace() {
                //if it is not a special character then we are going to select a substring whose end will be at :
                //--the one before the next following special character
                //--or the one before a whitespace
                //--or the one before the end of the sentence.
                //then we are going to determine the substring to be selected based on this comparision.
                for back in front+1..max_index {
                    if is_special(self.string.chars().nth(back).unwrap()) || self.string.chars().nth(back).unwrap().is_whitespace() || back == max_index-1 {
                        self.back = back;
                        return Some(&input_string[front..back]);
                    }
                }
            } else {
                self.back += 1;
            }
        }

        None

    }
}


impl Tokens {

    pub fn new() -> Tokens {
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
        let mut sparse: Vec<Vec<usize>> = vec![vec![0_usize ; frame.number_of_samples.try_into().unwrap()]];
        let mut mapper: HashMap<&str, usize> = HashMap::new();

        match &frame.data[column_index] {
            data_type::Strings(temp) => {
                //for each sentence in the given column.
                for (i , sentence) in temp.iter().enumerate() {
                    let lower_temp = sentence.to_lowercase();
                    let special_string: SpecialStr = SpecialStr::new(&lower_temp);
                    //comparing each word after making it lowercase.
                    //going through the special iterator ;)
                    for word in special_string.into_iter() {
                        //if the word is already occupied then we are going to just add 1 to the token distribution at that index
                        //or we are going to insert this value with the value 1. 
                        match self.column_index.entry(word.to_owned()) {
                            Entry::Occupied(temp) => {
                                let &index_of = temp.get();//where do we need to locate the word. 
                                token_distribution[index_of] += 1;
                                sparse[index_of][i] += 1;
                            },
                            Entry::Vacant(mut entry) => {
                                entry.insert(count);
                                token_distribution.push(1);
                                sparse.push(vec![0_usize ; frame.number_of_samples.try_into().unwrap()]);
                                sparse[count][i] += 1;
                                count += 1;
                            },
                        }
                    }
                
                }
            },
            _ => panic!("You cannot tokenise the float or the category data type"),
        } 
    }


}