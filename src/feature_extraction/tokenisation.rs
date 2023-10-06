//!FEATURE EXTRACTION:
//!Extract the wanted attributes from the data.

use std::borrow::Cow;
use std::collections::{HashMap, hash_map::Entry};
use fastrand::char;
use rayon::prelude::IntoParallelRefIterator;
use sprs::CsVec;

use crate::data_frame::{data_type::data_type, data_frame::data_frame};

pub struct Tokens {
    column_index: HashMap<String, sparse_vec_with_count>
}

#[derive(Debug)]
//we need to convert into some vec of vec of f32 when training but we will decrease the number of tokens by that time. 
struct sparse_vec_with_count {
    count : u32,//frequency of the item.
    sparse_vector: CsVec<u32>,
}

impl sparse_vec_with_count {
    fn distribution(&self) -> u32 {
        self.count
    }

    //really inefficient , just use for debugging.
    fn len(&self) -> usize {
        self.sparse_vector.iter().count()
    }
}


//special whitespace and special character dividing iterator.
pub struct SpecialStr<'a> {
    string: &'a str,
    back: usize,//index of the back of the &str substring.(we are using the charindices so we are for sure going to get an index which is a boundary)
}

impl<'a> SpecialStr<'a> {
    pub fn new(input : &'a str) -> Self {
        SpecialStr {string: input, back: 0}
    }
}

//to split at all the places which are special.
//but we need to give some special importance to the '?', '!', '', ''

//anything which is not a alphanumeric or a whitespace.
pub fn is_special(c: char) -> bool {
    !c.is_ascii_alphanumeric() && !c.is_whitespace()
}

//this iterator divides into "I am an assHoLe!." into an iterator which gives out ("i" , "am", "an" , "asshole" , "!", ".")
//we are going to split at the whitespaces and any special characters.


//TODO = we also probably need some variation of this which return the consecutive special chars as a single substring.'cause generally 

impl<'a> Iterator for SpecialStr<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let max_index = self.string.len();

        //the char indices iterator returns the byte position of each character and the character itself
        //so it can give out values like 23 ,27 , 28 consecutively which is not a problem , but the skip doesn't
        //care about it it just looks for the first n elements so it doesn't care about the byte index.
        for (i , character) in self.string.char_indices().skip(self.back) {
            //if the present char is a special character just return the character by itself.
            if is_special(character) {
                self.back += 1;
                return Some(&self.string[i..i+character.len_utf8()]);
            } else if !character.is_whitespace() {
                //if it is not a special character then we are going to select a substring whose end will be at :
                //--the one before the next following special character
                //--or the one before a whitespace
                //--or the one before the end of the sentence.
                //then we are going to determine the substring to be selected based on this comparision.
                for (back , character_2) in self.string.char_indices().skip(self.back+1) {
                    self.back += 1;
                    if is_special(character_2) || character_2.is_whitespace() || back+1 == max_index {
                        if back+1 == max_index && !is_special(self.string.chars().last().unwrap()) {
                            return Some(&self.string[i..max_index]);
                        }
                        return Some(&self.string[i..back]);
                    }
                }
            } else {//if there are more than 1 consecutive spaces
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
        }
    }

    //possible only for the string data type.
    pub fn tokenise(&mut self, frame : &data_frame, index : usize) {

        //not preallocating the memory because we do not know the number of individual words we are going to come across.
        let mut token_distribution: Vec<usize> = vec![];
        let mut count: usize = 0_usize;
        

        match &frame.data[index] {
            data_type::Strings(temp) => {
                let column_index = &mut self.column_index;
                //for each sentence in the given column.
                temp.iter().enumerate().for_each(|(string_index , sentence)| {
                    let lower_temp = sentence.to_lowercase();
                    let special_string: SpecialStr = SpecialStr::new(&lower_temp);
                    //comparing each word after making it lowercase.
                    //going through the special iterator ;)-
                    for word in special_string.into_iter() {
                        //if the word is already occupied then we are going to just add 1 to the token distribution at that index
                        //or we are going to insert this value with the value 1.
                        match column_index.entry(word.to_owned()) {
                            //if we already came across this word
                            Entry::Occupied(mut temp) => {
                                let to_mod = temp.get_mut();
                                to_mod.count+=1;
                                //we are calling unwrap because there is no way the indices vector is empty even after we got here.
                                if *to_mod.sparse_vector.indices().last().unwrap() == string_index {
                                    //this still takes log time but couldn't find a better way.
                                    match to_mod.sparse_vector.get_mut(string_index) {
                                        Some(temp) => *temp += 1,
                                        None => (),
                                    }
                                } else {
                                    to_mod.sparse_vector.append(string_index , 1);
                                }
                            },
                            //a new word!(i am not that exited tbh, that is just a word of expression, oh you are making me sad :( )
                            Entry::Vacant(mut entry) => {
                                entry.insert(sparse_vec_with_count {count: 1, sparse_vector: CsVec::new(frame.number_of_samples as usize , vec![string_index] , vec![1])});
                            },
                        }
                    }
                });
            },
            _ => panic!("You cannot tokenise the float or the category data type"),
        }
    }

    

    pub fn give_names(&self) {
        let temp = self.column_index.get(".").unwrap();
        print!("{}", temp.count);
    }


}