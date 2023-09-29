use crate::file_handling::read_from::read_csv;

use super::tokenisation::{self, Tokens};

#[cfg(test)]


#[test]
fn divide_n_print() {
    use super::tokenisation::SpecialStr;

    let input = "happy bday. .i. bevibae %#(%# vev nw .";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();
    print!("{:?}", temp);
    assert_eq!(temp , vec!["happy", "bday", ".", ".", "i", ".", "bevibae", "%", "#", "(", "%", "#", "vev", "nw", "."]);
}

#[test]
fn divide_n_print_2() {
    use super::tokenisation::SpecialStr;

    let input = "hvcvk (vuk == gvyv'' jvv( hvktvk";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    //println!("{:?}", temp);

    assert_eq!(temp , vec!["hvcvk", "(", "vuk", "=", "=", "gvyv", "'", "'", "jvv", "(", "hvktv"]);
}

#[test]

fn opening_and_tokenising() {
    let new_ = read_csv("C:/Users/HARSHA/Downloads/archive/test.csv", true, false).unwrap();
    
    
    new_.describe();
    
    let mut temp = Tokens::new();

    temp.tokenise(&new_, 1); 
}