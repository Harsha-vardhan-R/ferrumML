//!FEATURE EXTRACTION:
//!Extract the wanted attributes from the data.
//! >Tokenise the strings
//! >Manipulate them as wanted.
//! >Stemming/lemmatising the tokens.(rust-stemmers)
//! >Weighting the importance of the tokens using different algorithms like (tf-idf , BM-25 etc...)
//! >Turning this into a vec<vec<f32>> for training some model.


use crate::feature_extraction::tokenisation::special_iterator::is_special;
use std::collections::{HashMap, hash_map::Entry};
use fastrand::char;
use sprs::CsVec;
use crate::data_frame::{data_type::DataType, data_frame::DataFrame};
use rust_stemmers::{Algorithm , Stemmer};

use self::special_iterator::{SpecialStrings, SpecialStr, SpecialStrClump, SpecialStrDivideall};

pub struct Tokens {
    pub column_index: HashMap<String, SparseVecWithCount>
}


#[derive(Debug)]
//we need to convert into some vec of vec of f32 when training but we will decrease the number of tokens by that time. 
pub struct SparseVecWithCount {
    index : usize,//to be used to make the input and the output vectors.
    count : u32,//frequency of the item.
    sparse_vector: CsVec<u32>,
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


pub mod special_iterator {
    use std::char::from_digit;



    #[derive(Debug)]
    pub enum SpecialStrings<'a> {
        DivideSpecial(SpecialStr<'a>),//treats all the speacial charecters as different.
        ClumpSpecial(SpecialStrClump<'a>),//treats consecutive special characters as a single token.
        DivideAll(SpecialStrDivideall<'a>),//returns an iterator that returns each char except the whitespces.
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

}


impl Tokens {

    pub fn new() -> Tokens {
        Tokens {
            column_index: HashMap::new(),
        }
    }

    ///Tokenise a certain column of the data_frame
    ///possible only for the string data type.
    /// currently imple types for iterator types:
    ///->     "divide_special"
    ///->     "clump_special"
    ///->     "divide_all"
    pub fn tokenise(&mut self, frame : &DataFrame, index : usize, iterator_type : &str) {

        let mut index_here = 0;

        //not preallocating the memory because we do not know the number of individual words we are going to come across.
        let mut token_distribution: Vec<usize> = vec![];
        let mut count: usize = 0_usize;
        

        match &frame.data[index] {
            DataType::Strings(temp) => {
                let column_index = &mut self.column_index;
                //for each sentence in the given column.
                temp.iter().enumerate().for_each(|(string_index , sentence)| {
                    let lower_temp = sentence.to_lowercase();

                    let special_string: SpecialStrings = match iterator_type {
                        "divide_special" => SpecialStrings::DivideSpecial(SpecialStr::new(&lower_temp)),
                        "clump_special" => SpecialStrings::ClumpSpecial(SpecialStrClump::new(&lower_temp)),
                        "divide_all" => SpecialStrings::DivideAll(SpecialStrDivideall::new(&lower_temp)),
                        _ => panic!("no token iterator found with this name"),
                    };

                    count += 1;
                    
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
                            Entry::Vacant(mut entry) => {
                                entry.insert(SparseVecWithCount {index : index_here , count: 1, sparse_vector: CsVec::new(frame.number_of_samples as usize , vec![string_index] , vec![1])});
                                index_here += 1;
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

    ///Returns the number of times an individual token appears in all the input strings.
    pub fn get_count(&self , token_name : &str) -> u32 {
        let temp = self.column_index.get(token_name);
        match temp {
            Some(temp) => temp.count,
            None => 0,
        }
    }

    //function to delete.
    pub fn temp(&self) {
        for (index , element) in self.column_index.iter().enumerate() {
            eprint!("{:?}", element.0);
            if index > 1000 {
                break;
            }
        }
    }

    ///returns the number of tokens that occur less than or equal to the number of times.
    /// doesn't care about the frequency of appearing in different strings.
    pub fn count_below(&self , threshold  : u32) -> usize {
        let mut count = 0;
        
        let keys : Vec<String> = self.column_index.iter().filter(|(_ , value)| value.count <= threshold).map(|(key , _)| key.clone()).collect();
        count = keys.len();

        eprintln!(">Found {} values occuring less than {} times.", count , threshold);

        return count;
    }

    ///removes the tokens which occur less than r equal to a certain threshold number of times all togather.
    pub fn remove_weightless(&mut self, threshold  : u32) {

        let count: usize;
        
        let keys : Vec<String> = self.column_index.iter().filter(|(_ , value)| value.count <= threshold).map(|(key , _)| key.clone()).collect();
        count = keys.len();

        for key in keys {
            //eprint!("'{}'", key);
            self.column_index.remove(&key);
        }

        eprintln!(">Found {} values occuring less than or equal to {} times, Removed.", count , threshold);

    }

    ///removes all the tokens that have a special charater.
    /// threshold controls the size of the length of the strings,
    /// having 3 as the threshold value removes all the special tokens whose size is more than 3.
    /// 0 threshold removes all the tokens that start with ascii special characters.
    pub fn remove_special(&mut self, threshold  : usize) {

        let count: usize;
         
        let keys : Vec<String> = self.column_index.iter()
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
            self.column_index.remove(&key);
        }

        eprintln!(">Found {} special values having size less than or equal to {} times, Removed.", count , threshold);

    }


    ///using the 'rust-stemmers' library, this stems the strings if possible.
    ///storing the 'ing' 'ed' etc.. is optional.
    ///you can create an exception for this based on certain endings.
    fn stemm_tokens(&mut self, exception_vector : Vec<&str>) {

        let stemmer = Stemmer::create(Algorithm::English);
        let mut new_tokens = 0;
        let mut exist_tokens = 0;

        'outer: for (string , sparse) in self.column_index.iter() {
            for exception in exception_vector.iter() {
                if string.ends_with(exception) {
                    continue 'outer;
                }
            }

            let changed = stemmer.stem(&string);
            let changed_Str: &str = &changed;
            //we are going to store the names of the tokens cause we cannot iterate and remove at the same time.
            //if the value changed.
            if changed_Str != string {
                //if the value already exists in the hashmap, we are going to update it with the new stuff.
                if self.column_index.contains_key(changed_Str) {
                    //w somehow need to mix both the vectors with the repeated count.
                    let temp = self.column_index.get(changed_Str).unwrap();

                } else { // create a new token in the hashmap and fill it accordingly.

                }
            }
        }
    }


    

    //wiki def : A formula that aims to define the importance of a keyword or phrase within a document or a web page.
    
    fn weight_terms(&mut self, target_index : usize, weight_scheme : &str) {

    }

}

//function takes two inputs in and returns one jointed output.
/* fn rearrange_new(to_add1 : &SparseVecWithCount, to_add2 : &SparseVecWithCount) {
    let present_in_2 = 0;
    let mut new_swc = SparseVecWithCount{
        count : to_add1.count + to_add2.count,
        sparse_vector : CsVec::empty(1),
    };

    //let mut to_fill = CsVec::new(10 , vec![0_]);

    let mut iter1 = to_add1.sparse_vector.iter();
    let mut iter2 = to_add2.sparse_vector.iter();

    //we are going to call next and append that value to the neewly created hashmap.
    


} */