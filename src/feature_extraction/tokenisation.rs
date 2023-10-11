//!FEATURE EXTRACTION:
//!Extract the wanted attributes from the data.
//! Mainly tokenisation, token importance ranking.


use std::collections::{HashMap, hash_map::Entry};
use fastrand::char;
use rayon::prelude::IntoParallelRefIterator;
use sprs::CsVec;
use crate::data_frame::{data_type::DataType, data_frame::DataFrame};

pub struct Tokens {
    column_index: HashMap<String, SparseVecWithCount>
}


#[derive(Debug)]
//we need to convert into some vec of vec of f32 when training but we will decrease the number of tokens by that time. 
struct SparseVecWithCount {
    count : u32,//frequency of the item.
    sparse_vector: CsVec<u32>,
    TfIdf : Option<Vec<usize>>
}

impl SparseVecWithCount {

    fn distribution(&self) -> u32 {
        self.count
    }

    //really inefficient , just use for debugging.
    //returns the number of strings in which this token exists.
    //doesn't care about the frequency.
    fn len(&self) -> usize {
        self.sparse_vector.iter().count()
    }

}

#[derive(Debug)]
enum SpecialStrings<'a> {
    DivideSpecial(SpecialStr<'a>),//treats all the speacial charecters as different.
    ClumpSpecial(SpecialStrClump<'a>)//treats consecutive special characters as a single token.
}

impl<'a> IntoIterator for SpecialStrings<'a> {
    type Item = &'a str;
    type IntoIter = Box<dyn Iterator<Item = &'a str> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            SpecialStrings::DivideSpecial(special_str) => {
                Box::new(special_str.into_iter())
            }
            SpecialStrings::ClumpSpecial(special_str_clump) => {
                Box::new(special_str_clump.into_iter())
            }
        }
    }

}



#[derive(Debug)]
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

#[derive(Debug)]
pub struct SpecialStrClump<'a> {
    string: &'a str,
    back: usize,
}

impl<'a> SpecialStrClump<'a> {
    pub fn new(input : &'a str) -> Self {
        SpecialStrClump {string: input, back: 0}
    }
}

//to split at all the places which are special.
//but we need to give some special importance to the '?', '!', '', ''
//anything which is not a alphanumeric or a whitespace.
pub fn is_special(c: char) -> bool {
    !c.is_ascii_alphanumeric() && !c.is_whitespace()
}

//this iterator divides into "I am an assHoLe!. .." into an iterator which gives out ("i" , "am", "an" , "assHoLe" , "!", ".", ".", ".")(they won't be lowercase , but that happens in the tokeniser)
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
                        //IMP : do not get confused back is a valid character end unlike self.back, which is the count of the nth character
                        //sorry for the bad naming.
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

//this iterator divides into "I am an assHoLe!. .." into an iterator which gives out ("i" , "am", "an" , "assHoLe" , "!.", "..").the turning this into lowercase is done in the tokeniser.
//we are going to split at the whitespaces and any special character boundaries.

impl<'a> Iterator for SpecialStrClump<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let max_index = self.string.len();

        //the char indices iterator returns the byte position of each character and the character itself
        //so it can give out values like 23 ,27 , 28 consecutively which is not a problem , but the skip doesn't
        //care about it it just looks for the first n elements so it doesn't care about the byte index.
        for (i , character) in self.string.char_indices().skip(self.back) {
            //if the present char is a special character just return the character by itself.
            if is_special(character) {
                //self.back += 1;
                for (back , character_2) in self.string.char_indices().skip(self.back+1) {
                    self.back += 1;
                    if back+1 == max_index {
                        return Some(&self.string[i..max_index]);
                    } else if is_special(character_2) {
                        continue;
                    } 
                    return Some(&self.string[i..back]);
                }
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

    ///Tokenise a certain column of the data_frame
    ///possible only for the string data type.
    pub fn tokenise(&mut self, frame : &DataFrame, index : usize, clump_special : bool) {

        //not preallocating the memory because we do not know the number of individual words we are going to come across.
        let mut token_distribution: Vec<usize> = vec![];
        let mut count: usize = 0_usize;
        

        match &frame.data[index] {
            DataType::Strings(temp) => {
                let column_index = &mut self.column_index;
                //for each sentence in the given column.
                temp.iter().enumerate().for_each(|(string_index , sentence)| {
                    let lower_temp = sentence.to_lowercase();

                    let special_string: SpecialStrings = match clump_special {
                        false => SpecialStrings::DivideSpecial(SpecialStr::new(&lower_temp)),
                        true => SpecialStrings::ClumpSpecial(SpecialStrClump::new(&lower_temp))
                    };
                    
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
                                entry.insert(SparseVecWithCount {count: 1, sparse_vector: CsVec::new(frame.number_of_samples as usize , vec![string_index] , vec![1]) , TfIdf : None});
                            },
                        }
                    }
                });
            },
            _ => panic!("You cannot tokenise the float or the category data type"),
        }
    }

    ///to get the statistics about the tokens.
    pub fn get_stats(&self) {
        println!("Total number of tokens : {}", self.column_index.len());


    }

    //wiki def (tf-idf) : A formula that aims to define the importance of a keyword or phrase within a document or a web page.




    ///Returns the number of times an individual token appears in all the input strings.
    pub fn get_count(&self , token_name : &str) -> u32 {
        let temp = self.column_index.get(token_name).unwrap();
        temp.count
    }

}