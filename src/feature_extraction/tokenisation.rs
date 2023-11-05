//!FEATURE EXTRACTION:
//!Extract the wanted attributes from the data.
//! >Tokenise the strings
//! >Manipulate them as wanted.
//! >Stemming/lemmatising the tokens.(rust-stemmers)
//! >Weighting the importance of the tokens using different algorithms like (tf-idf , BM-25 etc...)
//! >Turning this into a vec<vec<f32>> for training some model.


use crate::feature_extraction::tokenisation::special_iterator::is_special;
use std::{collections::{HashMap, hash_map::Entry}};
use sprs::CsVec;
use crate::data_frame::{data_type::DataType, data_frame::DataFrame};
use rust_stemmers::{Algorithm , Stemmer};
use self::special_iterator::{SpecialStrings, SpecialStr, SpecialStrClump, SpecialStrDivideall, SpeciaStrDivideCustom};

pub struct Tokens {
    ///HashMap<Token , (Index, Count)>
    pub token_map_index: HashMap<String, (usize , usize)>,
    ///HashMap<Index , &token(reference to the token)>
    pub index_map_token: HashMap<usize, String>,
    ///Store the sequence of each sentence in an encoded form.
    pub data_in_sequence: Vec<Vec<usize>>,
}

impl Tokens {

    ///Creating a new token object, need to have the number of samples to pre-allocate the size.
    ///`DataFrame.number_of_sample` gives you the number of samples as u32, please cast it before using in this function.
    pub fn new(number_of_samples : usize) -> Tokens {
        Tokens {
            token_map_index: HashMap::new(),
            index_map_token: HashMap::new(),
            data_in_sequence: vec![vec![] ; number_of_samples]
        }
    }

}


pub mod special_iterator {
    use std::collections::HashMap;


    ///An enum that stores different types of special iterator types.
    #[derive(Debug)]
    pub enum SpecialStrings<'a> {///treats all the speacial charecters as different.
        DivideSpecial(SpecialStr<'a>),///treats consecutive special characters as a single token.
        ClumpSpecial(SpecialStrClump<'a>),///returns an iterator that returns each char except the whitespces.
        DivideAll(SpecialStrDivideall<'a>),///returns an iterator which divides the input string at de-limiters which user provides, for multilingual tokenization.
        DivideCustomSpaces(SpeciaStrDivideCustom<'a>),
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
                SpecialStrings::DivideAll(special_str_divide_all) => {
                    Box::new(special_str_divide_all.into_iter())
                }
                SpecialStrings::DivideCustomSpaces(special_str_divide_custom) => {
                    Box::new(special_str_divide_custom.into_iter())
                }
            }
        }

    }


    //###########################################
    #[derive(Debug)]
    //special whitespace and special character dividing iterator.
    //we should not duplicate the data anywhere.
    //we are going to
    pub struct SpecialStr<'a> {
        string: &'a str,
        //this is going to store the start of the present token
        //it "SHOULD" be a char boundary, we are going to make sure of 
        //that in the iterator impl.
        front: usize, 
    }

    impl<'a> SpecialStr<'a> {
        pub fn new(input : &'a str) -> Self {
            SpecialStr {
                string: input,
                //initially starting at 0.
                front: 0
            }
        }
    }

    //##############################################
    #[derive(Debug)]
    pub struct SpecialStrClump<'a> {
        string: &'a str,
        front: usize,
    }

    impl<'a> SpecialStrClump<'a> {
        pub fn new(input : &'a str) -> Self {
            SpecialStrClump {string: input, front: 0}
        }
    }

    //##########################################
    #[derive(Debug)]
    pub struct SpecialStrDivideall<'a> {
        string: &'a str,
        front: usize,
    }

    impl<'a> SpecialStrDivideall<'a> {
        pub fn new(input : &'a str) -> Self {
            SpecialStrDivideall {string: input, front: 0}
        }
    }

    //##########################################
    #[derive(Debug)]
    pub struct SpeciaStrDivideCustom<'a> {
        string: &'a str,
        front: usize,
        special_divide: HashMap<char , usize>,//hashmap to make comparision faster.. and get the length(reason for not using a hashset)
    }

    ///Delimit at any UTF-8 character.
    impl<'a> SpeciaStrDivideCustom<'a> {
        ///delimiters can be any UTF-8 character.
        pub fn new(input : &'a str , delimiters : Vec<char>) -> Self {
            let mut hashyy = HashMap::new();
            
            for delimiter in delimiters {
                hashyy.insert(delimiter, delimiter.len_utf8());
            }

            SpeciaStrDivideCustom { 
                string: input,
                front: 0,
                special_divide : hashyy,
            }

        }
    }

    //#####################
    //SPECIAL ITERATORS
    //#####################


    //to split at all the places which are special.
    //but we need to give some special importance to the '?', '!'.
    //anything which is not a alphanumeric or a whitespace.
    //but this means we only consider the english letters and all the other language letters will be divided into individual or clumped. 
    pub fn is_special(c: char) -> bool {
        !c.is_ascii_alphanumeric() && !c.is_whitespace()//try adding !c.is_ascii() , we can tokenise even other languages properly
    }

    //########WORKING AS EXPECTED#############
    //this iterator divides into "I am an assHoLe!. .." into an iterator which gives out ("i" , "am", "an" , "assHoLe" , "!", ".", ".", ".")(they won't be lowercase ,that happens in the tokeniser function)
    //we are going to split at the whitespaces and any special characters.
    //the divide_special iterator. 
    impl<'a> Iterator for SpecialStr<'a> {
        type Item = &'a str;

        fn next(&mut self) -> Option<Self::Item> {
            //we are iterating only on the remaining part of the string for a linear time complexity.
            let mut bound_index : usize = self.front;

            for char_1 in self.string[self.front..].chars() {
                if is_special(char_1) {//update the self.front and return the special character alone.
                    self.front = bound_index+char_1.len_utf8();//this is a guaranteed char boundary.
                    return Some(&self.string[bound_index..self.front]);
                } else if !char_1.is_whitespace() {//ascii whitespace also matches to newline and a couple of other characters
                    for (back_byte_index , char_2) in self.string[bound_index..].char_indices() {
                        if is_special(char_2) || char_2.is_whitespace() {
                            self.front = bound_index+back_byte_index;
                            return Some(&self.string[bound_index..self.front]);                            
                        }
                    }
                    self.front = self.string.len();
                    return Some(&self.string[bound_index..]);//if there is nothing left in the above iterator.
                } else {
                    bound_index += char_1.len_utf8();
                } 
            }
            return None;//this is going to be the last return for the lazy iterator.
        }
    }

    //######################WORKING AS EXPECTED#############
    //this iterator divides into "I am an assHoLe!. .." into an iterator which gives out ("i" , "am", "an" , "assHoLe" , "!.", "..").the turning this into lowercase is done in the tokeniser.
    //we are going to split at the whitespaces and any special character boundaries.
    impl<'a> Iterator for SpecialStrClump<'a> {
        type Item = &'a str;

        fn next(&mut self) -> Option<Self::Item> {
            let mut bound_index : usize = self.front;

            for char_1 in self.string[self.front..].chars() {
                if is_special(char_1) {//update the self.front and return the special character alone.
                    for (back_byte_index , char_2) in self.string[bound_index..].char_indices() {
                        if !is_special(char_2) || char_2.is_whitespace() {//finding delimiter after encountering a non special character.
                            self.front = bound_index+back_byte_index;
                            return Some(&self.string[bound_index..self.front]);                            
                        }
                    }
                    self.front = self.string.len();
                    return Some(&self.string[bound_index..]);//if there is nothing left in the above iterator.
                } else if !char_1.is_whitespace() {//ascii whitespace also matches to newline and a couple of other characters
                    for (back_byte_index , char_2) in self.string[bound_index..].char_indices() {
                        if is_special(char_2) || char_2.is_whitespace() {//finding delimiter after encountering a non special character.
                            self.front = bound_index+back_byte_index;
                            return Some(&self.string[bound_index..self.front]);                            
                        }
                    }
                    self.front = self.string.len();
                    return Some(&self.string[bound_index..]);//if there is nothing left in the above iterator.
                } else {
                    bound_index += char_1.len_utf8();
                } 
            }
            return None;        
        }

    }


    //#################WORKING AS EXPECTED############################
    impl<'a> Iterator for SpecialStrDivideall<'a> {
        type Item = &'a str;

        fn next(&mut self) -> Option<Self::Item> {
            for i in self.string[self.front..].chars() {
                if !i.is_ascii_whitespace() {
                    self.front += i.len_utf8();
                    return Some(&self.string[self.front-i.len_utf8()..self.front]);
                } else {
                    self.front += i.len_utf8();
                }
            }
            return None;
        }
    }

    //#################WORKING AS EXPECTED#############################
    impl<'a> Iterator for SpeciaStrDivideCustom<'a> {
        type Item = &'a str;

        fn next(&mut self) -> Option<Self::Item> {
            let mut bound_index : usize = self.front;
            let max_ = self.string.len();

            for char_1 in self.string[self.front..].chars() {
                match self.special_divide.get(&char_1) {
                    Some(length) => {
                        bound_index += *length;//if the hashmap contains the character we are just going to skip it.
                    },
                    None => {
                        for (front_here , char_2) in self.string[bound_index..].char_indices() {
                            if self.special_divide.contains_key(&char_2) {
                                self.front = bound_index+front_here;
                                return Some(&self.string[bound_index..self.front]);
                            }
                        }
                        self.front = self.string.len();
                        return Some(&self.string[bound_index..]);
                    },
                }
            }
            return None;
        }

    }

}


impl Tokens {

    /// Tokenise a certain column of the data_frame
    /// possible only for the string data type.
    /// 
    /// definition of a special character - `!c.is_ascii_alphanumeric() && !c.is_whitespace()`.(not used in the custom_delimit iterator)
    /// 
    /// `custom_delimiters` will not be considered for any other iterator_type other than custom_delimit.
    /// 
    /// currently imple types for iterator types:
    /// 
    /// `divide_special` `id = 1` - tokens after each word is divided at white spaces and special chaacters consecutive special charaters will be treated as individual,
    /// 
    /// `clump_special` `id = 2` - tokens after each word is divided at white spaces and special chaacters consecutive special charaters will be treated as a single token until terminated by a whitespace or a normal character
    /// 
    /// `divide_all` `id = 3` - each charater will be treated as an individual token indipendent of being special or not.
    /// 
    /// `custom_delimit` `id = 4` - characters are divided according to user input delimiting characters.
    /// 
    /// Memory consumptive.
    pub fn tokenise(&mut self, frame : &DataFrame, index : usize, iterator_type_id : usize, custom_delimiters : Option<Vec<char>>) {
        let start_time = std::time::Instant::now();

        match &frame.data[index] {
            DataType::Strings(temp) => {
                let column_index = &mut self.token_map_index;
                //to keep track of encoding of the present token;
                let mut present_index = 0_usize;
                
                //for each sentence in the given column.
                temp.iter().enumerate().for_each(|(string_index , sentence)| {

                    let lower_temp = sentence.to_lowercase();

                    let special_string: SpecialStrings = match iterator_type_id {
                        1 => SpecialStrings::DivideSpecial(SpecialStr::new(&lower_temp)),
                        2 => SpecialStrings::ClumpSpecial(SpecialStrClump::new(&lower_temp)),
                        3 => SpecialStrings::DivideAll(SpecialStrDivideall::new(&lower_temp)),
                        4 => SpecialStrings::DivideCustomSpaces(SpeciaStrDivideCustom::new(&lower_temp, custom_delimiters.clone().expect("a `Some(Vec<char>)` is expected."))),
                        _ => panic!("no token iterator found with this id"),
                    };
                    
                    //comparing each word after making it lowercase.
                    //going through the special iterator.
                    for token in special_string.into_iter() {
                        match self.token_map_index.entry(token.to_string()) {
                            Entry::Occupied(mut occupied_value) => {
                                let index_of_token = occupied_value.get_mut();
                                self.data_in_sequence[string_index].push(index_of_token.0);
                                index_of_token.1 += 1;
                            }
                            Entry::Vacant(mut empty_value) => {
                                empty_value.insert((present_index , 1));//insert the new value into `token_map_index` with these values.
                                self.index_map_token.insert(present_index, token.to_owned());
                                self.data_in_sequence[string_index].push(present_index);
                                present_index += 1;
                            },
                        }
                    }

                });
            },
            _ => panic!("You cannot tokenise the float or the category data type"),
        }

        eprintln!("> Tokenised {} Strings with {} unique tokens in : {:?}", frame.number_of_samples, self.index_map_token.len(), start_time.elapsed());

    }

    ///to get the statistics about the tokens.
    pub fn get_stats(&self) {
        println!("Total number of tokens : {}", self.token_map_index.len());
    }

    ///Returns the number of times an individual token appears in all the input strings.
    pub fn get_count(&self , token_name : &str) -> u32 { 
        let temp = self.token_map_index.get(token_name);
        match temp {
            Some(temp) => temp.1 as u32,
            None => 0,
        }
    }

    ///returns the number of tokens that occur less than or equal to the number of times.
    /// doesn't care about the frequency of appearing in different strings.
    pub fn count_below(&self , threshold  : usize) -> usize {
        let mut count = 0;
        let keys : Vec<String> = self.token_map_index.iter().filter(|(_ , value)| value.1 <= threshold).map(|(key , _)| key.clone()).collect();
        count = keys.len();
        eprintln!(">Found {} values occuring less than {} times.", count , threshold);
        return count;
    }


    ///removes the tokens which occur less than r equal to a certain threshold number of times all togather.
    pub fn remove_sparse_tokens(&mut self, threshold  : usize) {

        let count: usize;
        
        let keys : Vec<String> = self.token_map_index.iter().filter(|(_ , value)| value.0 <= threshold).map(|(key , _)| key.clone()).collect();
        count = keys.len();

        for key in keys {
            let index_to_remove = self.token_map_index.remove(&key).unwrap().0;//removing and getting the index from the token_to_index.
            self.index_map_token.remove(&index_to_remove);
        }

        eprintln!(">Found {} values occuring less than or equal to {} times, Removed.", count , threshold);

    }

    ///removes all the tokens that have a special charater.
    /// threshold controls the size of the length of the strings,
    /// having 3 as the threshold value removes all the special tokens whose size is more than 3.
    /// 0 threshold removes all the tokens that start with ascii special characters.
    pub fn remove_special(&mut self, threshold  : usize) {

        let count: usize;
         
        let keys : Vec<String> = self.token_map_index.iter()
            .filter(|(token , _)| (
                is_special(token.chars().next().unwrap_or_default())) && 
                (token.is_ascii()) && 
                (token.len() > threshold)
            )
            .map(|(key , _)| key.clone())
            .collect();

        count = keys.len();

        for key in keys {
            //eprint!("'{}'", key);
            self.token_map_index.remove(&key);
        }

        eprintln!(">Found {} special values having size less than or equal to {} times, Removed.", count , threshold);

    }

    ///using the 'rust-stemmers' library, this stems the strings if possible.
    ///storing the 'ing' 'ed' etc.. is optional.
    ///you can create an exception for this based on certain endings.
    ///for example if you have an element `ing` int the exception_vector it does not care to do any thing with the tokens that end with `ing`.
    ///
    ///`remove_changed` - keep or remove the changed values.`
    /// 
    ///`replace_changed` ##OVERRIDES OTHER PARAMETERS## - if true, we are going to replace just the token with the new stemmed one IF it does not exist in the hashmap already.
    /// 
    /// This method is `ALWAYS` going to skip the tokens which are special. so, obviously does not work for any other languages other than english.(well, for now at least)
    /// 
    /// if replace_changed is false and remove_changed is true then any value whose stemmed value is not present in the hashmap will be removed.
    pub fn stemm_tokens(&mut self, exception_vector: Vec<&str>, remove_changed: bool, replace_changed: bool) {
        //A LOT OF CLONING HAPPENS HERE, "MAYBE" SHOULD BE IMPROVED IF POSSIBLE.
        //PLEASE FIND A WAY TO PARALLALISE THIS, PRESENTLY THIS IS EXTREMELY SLOW.

        let stemmer = Stemmer::create(Algorithm::English);
        let mut new_tokens = 0;
        let mut exist_tokens = 0;
        let mut presnt_index = 0_usize;
        let mut tokens: Vec<String> = vec![];
        for element in self.token_map_index.iter() {
            tokens.push(element.0.clone());
        }
        let mut buffer = vec![0_u32 ; self.data_in_sequence.len()];
        let mut removed_index: Vec<usize> = vec![];//stores the indexes of removed tokens.

        'outer: for token_name in tokens.iter() {

            for exception in exception_vector.iter() {
                if token_name.ends_with(*exception) {//to the next token.
                    continue 'outer;
                }
            }

            let changed = stemmer.stem(token_name);
            let changed_str: &str = &changed;

            if changed_str != token_name {//if the stemmed token is different from the unstemmed one.
                if let Some(mut sparse_here) = self.token_map_index.get_mut(changed_str).cloned() {// If the stemmed value exists in the hashmap
                    
                } else {
                    if replace_changed {// Replace the token name with the new, stemmed token name while leaving the rest of the data exactly the same
                        
                    } else if remove_changed {//if you want to remove the token that is stemmed and the stemmed value is not in the hashmap.
                        
                    }
                }
            } else {
                
            }

        }

    }

    ///wiki def : A formula that aims to define the importance of a keyword or phrase within a document or a web page.
    /// 
    ///this is for training the 
    fn weight_terms(&mut self, target_index : usize, weight_scheme : &str) {

    }

}

fn clean_buffer(buffer : &mut Vec<u32>) {
    for i in 0..buffer.len() {
        buffer[i] = 0;
    }
}

/* 
//function takes two inputs in and returns one jointed output as a csvec.
//the buffer MUST have all the elements to be 0_usize before being sent into this function.
fn rearrange_new(to_add1 : &SparseVecWithCount, to_add2 : &SparseVecWithCount, buffer : &mut Vec<u32>) -> sprs::CsVecBase<Vec<usize>, Vec<u32>, u32> {

    //this iter is really fast because it iterates only on non zero values.
    for (index_1 , element) in to_add1.sparse_vector.iter() {
        buffer[index_1] = *element;
    }

    for (index_1 , element) in to_add2.sparse_vector.iter() {
        buffer[index_1] += *element;
    }

    let mut new_csvec = CsVec::empty(buffer.len());

    for (index , &num) in buffer.iter().enumerate() {
        if !(num == 0) {
            new_csvec.append(index, num);
        }
    }

    return new_csvec;

}
*/

///for an individual string
pub fn stemm_string(lower_temp : &str, iterator_type : &str, exception_endings : Vec<&str>) -> Vec<String> {
    let stemmer = Stemmer::create(Algorithm::English);

    let mut output_vec: Vec<String> = vec![];

    let special_string: SpecialStrings = match iterator_type {
        "divide_special" => SpecialStrings::DivideSpecial(SpecialStr::new(&lower_temp)),
        "clump_special" => SpecialStrings::ClumpSpecial(SpecialStrClump::new(&lower_temp)),
        "divide_all" => SpecialStrings::DivideAll(SpecialStrDivideall::new(&lower_temp)),
        _ => panic!("no token iterator found with this name"),
    };

    'outer: for token in special_string.into_iter() {
        for ending in exception_endings.iter() {
            if token.ends_with(*ending) {
                continue 'outer;
            }
        }

        output_vec.push((&stemmer.stem(token)).to_string());
    }

    return output_vec;
}

///for removing the stop words, changes the input vector.
///this function expects you to give in the input vector which you got by collecting a vector of Strings after tokenising them.
///and the to_lowercase does exactly what it means-turns all the input strings to lowercase.
pub fn remove_stop_words(token_in_order : &mut Vec<String>, stop_words : &Vec<&str>) {
    let mut notedown_names: Vec<usize> = vec![];

    for (index, token) in token_in_order.iter().enumerate() {
        if stop_words.contains(&&token.as_str()) {
            notedown_names.push(index);
        }
    }

    for i in notedown_names.iter().rev() {
        token_in_order.remove(*i);
    }

}


//removes every token which is special in the given vector of tokens.
pub fn remove_special_words(token_in_order : &mut Vec<String>, stop_words : &Vec<&str>) {
    let mut notedown_names: Vec<usize> = vec![];

    for (index, token) in token_in_order.iter().enumerate() {
        if is_special(token.chars().nth(1).unwrap_or_default()) {
            notedown_names.push(index);
        }
    }

    for i in notedown_names.iter().rev() {
        token_in_order.remove(*i);
    }

}


