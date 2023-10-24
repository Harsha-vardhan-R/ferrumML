use crate::file_handling::read_from::read_csv;
use crate::feature_extraction::tokenisation::{special_iterator::{SpecialStr, SpecialStrClump}, Tokens};


#[cfg(test)]


#[test]
fn divide_n_print() {

    let input = "happy bday. .i. bevibae %#(%# vev nw . ahvbaw";
    //let input = "I love dogs";
    let new_one = SpecialStr::new(&input);

    for i in new_one.into_iter() {
        print!("{} | ", i);
    }

    println!();

    let new_one = SpecialStrClump::new(&input);

    for i in new_one.into_iter() {
        print!("{} | ", i);
    }
    //assert_eq!(temp , vec!["happy", "bday", ".", ".", "i", ".", "bevibae", "%", "#", "(", "%", "#", "vev", "nw", "."]);
}

#[test]
fn divide_n_print_2() {

    let input = "hvcvk (vuk == gvyv'' jvv( hvktvk!";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    println!("{:?}", temp);

    let new_one = SpecialStrClump::new(&input);

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
    let new_one = SpecialStrClump::new(&input);


    for i in new_one.into_iter() {
        println!("{}", i);
    } 

    //assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "½", "ay", "al"]);

}

#[test]
fn divide_n_print_5() {

    let input = "你好, 这是一个随机生成的中文UTF-8字符串。";
    let new_one = SpecialStrClump::new(&input);

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

    let mut new_ = read_csv(r#"testing_data/_archive/training.1600000.processed.noemoticon.csv"#, true, false).unwrap();
    //let start_time = std::time::Instant::now();

    //new_.set_headers(vec!["texthash", "text" , "selected_text" , "sentiment_target" ,"time", "age of user" , "country" , "population" , "area" , "density" ]);

    new_.describe();

    //new_.describe_the("text", false);

    //println!("Time taken to describe is : {:?}", start_time.elapsed());
    

    let mut temp = Tokens::new();
    let start_time = std::time::Instant::now();
    temp.tokenise(&new_, 5 , "clump_special");

    println!("Time taken to tokenise is : {:?}", start_time.elapsed());

    println!("'...' occurs {} times, total number of tokens is : {}", temp.get_count("..."), temp.column_index.len());
    println!("and that of '.' occurs is : {} times", temp.get_count("."));

    

}

#[test]



fn big_file_tokenise_benchmark() {
    let start_time = std::time::Instant::now();

    use std::io::Write;

    let new_text = std::fs::read_to_string(r#"testing_data/shake.txt"#).expect("this file does not even exist man!!");
    let special = SpecialStr::new(&new_text);

    let tokkk = special.into_iter().collect::<Vec<&str>>();

    println!("number of individual tokens is : {}" ,tokkk.len());

    println!("Time taken to tokenise and vectorise is : {:?}", start_time.elapsed());

    let mut new_file = std::fs::File::create(r#"testing_data/bake.txt"#).expect("couldn't create a new file");
    let strin = format!("{:?}", tokkk);
    let start_time = std::time::Instant::now();

    new_file.write_all(strin.as_bytes());

    println!("Time taken to write is : {:?}", start_time.elapsed());
    
}