use rayon::prelude::IndexedParallelIterator;

use crate::{file_handling::read_from::read_csv, feature_extraction::tokenisation::is_special};

use super::tokenisation::{self, Tokens};

#[cfg(test)]


#[test]
fn divide_n_print() {
    use super::tokenisation::SpecialStr;

    //let input = "happy bday. .i. bevibae %#(%# vev nw .";
    let input = "I love dogs";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();
    print!("{:?}", temp);
    //assert_eq!(temp , vec!["happy", "bday", ".", ".", "i", ".", "bevibae", "%", "#", "(", "%", "#", "vev", "nw", "."]);
}

#[test]
fn divide_n_print_2() {
    use super::tokenisation::SpecialStr;

    let input = "hvcvk (vuk == gvyv'' jvv( hvktvk!";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    println!("{:?}", temp);

    assert_eq!(temp , vec!["hvcvk", "(", "vuk", "=", "=", "gvyv", "'", "'", "jvv", "(", "hvktvk" , "!"]);
}

//still need to test this on social media emojies and other utf8 exclusives and stuff.

#[test]

////////this is not exactly working as it should, but just going with it.
fn divide_n_print_3() {
    use super::tokenisation::SpecialStr;

    let input = "` i love mine, too . happy motherï¿½s day to all";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();
    
    println!("{:?}",&temp);

    assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "¿", "½", "s", "day", "to", "all"]);


}

#[test]
fn divide_n_print_4() {
    use super::tokenisation::SpecialStr;

    let input = "Journey!? Wow... u just became cooler.  hehe... (is that possible!?)";
    let new_one = SpecialStr::new(&input);

    for i in new_one.into_iter() {
        println!("{}", i);
    }

    //assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "½", "ay", "al"]);

}
#[test]

fn opening_and_tokenising() {
    let new_ = read_csv("C:/Users/HARSHA/Downloads/archive/train.csv", true, false).unwrap();
    new_.describe();
    
    let mut temp = Tokens::new();
    let start_time = std::time::Instant::now();

    temp.tokenise(&new_, 1);

    println!("Time taken to tokenise is : {:?}", start_time.elapsed());
    temp.give_names();


}