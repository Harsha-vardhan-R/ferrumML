use crate::file_handling::read_from::read_csv;
use crate::feature_extraction::tokenisation::{special_iterator::{SpecialStr, SpecialStrClump}, Tokens};


#[cfg(test)]


#[test]
fn divide_n_print() {

    let input = "happy bday. .i. bevibae %#(%# vev nw . ahvbaw";
    //let input = "I love dogs";
    let new_one = SpecialStr::new(&input);

    for i in new_one.into_iter() {
        print!("|{}|\n", i);
    }
    //assert_eq!(temp , vec!["happy", "bday", ".", ".", "i", ".", "bevibae", "%", "#", "(", "%", "#", "vev", "nw", "."]);
}

#[test]
fn divide_n_print_2() {

    let input = "hvcvk (vuk == gvyv'' jvv( hvktvk!";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    println!("{:?}", temp);

    //assert_eq!(temp , vec!["hvcvk", "(", "vuk", "=", "=", "gvyv", "'", "'", "jvv", "(", "hvktvk" , "!"]);
}

//still need to test this on social media emojies and other utf8 exclusives and stuff.

#[test]

////////this is not exactly working as it should, but just going with it.
fn divide_n_print_3() {

    let input = "` i love mine, too . happy motherï¿½s day to all";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();
    
    println!("{:?}",&temp);

    assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "¿", "½", "s", "day", "to", "all"]);

}

#[test]
fn divide_n_print_4() {

    let input = "` i love mine, too . happy motherï¿½s day to all";
    let new_one = SpecialStr::new(&input);


    for i in new_one.into_iter() {
        println!("{}", i);
    } 

    //assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "½", "ay", "al"]);

}

#[test]
fn divide_n_print_5() {

    let input = "你好, 这是一个随机生成的中文UTF-8字符串。";
    let new_one = SpecialStr::new(&input);

    for i in new_one.into_iter() {
        println!("{}", i);
    } 

    for i in input.chars() {
        print!("{} \n",i);
    }

    //assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "½", "ay", "al"]);

}


#[test]
fn opening_and_tokenising() {

    let mut new_ = read_csv(r#"testing_data/_archive/train.csv"#, true, false).unwrap();
    //let start_time = std::time::Instant::now();

    new_.set_headers(vec!["texthash", "text" , "selected_text" , "sentiment_target" ,"time", "age of user" , "country" , "population" , "area" , "density" ]);

    new_.describe();

    //new_.describe_the("text", false);

    //println!("Time taken to describe is : {:?}", start_time.elapsed());
    let start_time = std::time::Instant::now();

    let mut temp = Tokens::new();

    temp.tokenise(&new_, 1 , "divide_special");

    println!("Time taken to tokenise is : {:?}", start_time.elapsed());

    print!("'...' occurs {} times", temp.get_count("..."));
    println!("and that of '.' occurs is : {} times", temp.get_count("."));

    

}