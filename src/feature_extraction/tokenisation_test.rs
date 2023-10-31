use std::io::Write;
use plotlib::repr::CategoricalRepresentation;
use crate::{feature_extraction::tokenisation::{special_iterator::{SpecialStr, SpecialStrClump, SpecialStrDivideall}, Tokens, stemm_string, remove_stop_words}, file_handling::read_from::read_csv};
use crate::feature_extraction::tokenisation::special_iterator::SpeciaStrDivideCustom;

#[cfg(test)]


#[test]
fn divide_n_print() {
    //let input = "happy bday. .i. bevibae %#(%# vev nw . ahvbaw";
    let input = "` i love mine, too . happy motherï¿½s day to all";
    //let new_one = SpeciaStrDivideCustom::new(&input , vec![' ' , '.' , ',' , '一']);
    let new_one = SpeciaStrDivideCustom::new(&input , vec![' ' , '¿']);

    for i in new_one.into_iter() {
        print!("{} |||| ", i);
    }

    //assert_eq!(temp , vec!["happy", "bday", ".", ".", "i", ".", "bevibae", "%", "#", "(", "%", "#", "vev", "nw", "."]);
}

#[test]
fn divide_n_print_2() {

    let input = "hvcvk (vuk == gvyv'' jvv( hvktvk!";
    let new_one = SpecialStrDivideall::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    println!("{:?}", temp);

    let new_one = SpecialStrClump::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    println!("{:?}", temp);

    //assert_eq!(temp , vec!["hvcvk", "(", "vuk", "=", "=", "gvyv", "'", "'", "jvv", "(", "hvktvk" , "!"]);
}

//still need to test this on social media emojies and other utf8 exclusives and stuff.

#[test]


fn divide_n_print_3() {

    let input = "` i love mine, too . happy motherï¿½s day to all";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();
    
    //println!("{:?}",&temp);

    assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "¿", "½", "s", "day", "to", "all"]);

}

#[test]
fn divide_n_print_4() {

    let input = "` i love mine, too . happy motherï¿½s day to all";
    let new_one = SpecialStr::new(&input);


    let temp = new_one.into_iter().map(|a_str| a_str.to_string()).collect::<Vec<String>>();

    assert_eq!(temp , vec!["`", "i", "love", "mine", ",", "too", ".", "happy", "mother", "ï", "¿", "½", "s", "day", "to", "all"]);

}

#[test]
fn divide_n_print_5() {//for different languages.
    //let input = "你好, 这是一个随机生成的中文UTF-8字符串。";
    //let input = "हिंदी भाषा एक भारतीय भाषा है और इसका लिपि देवनागरी है। यह विशेष रूप से भारत, नेपाल, और दुनियाभर में बोली जाती है। हिंदी में कई बेहतरीन काव्य और साहित्य के रचयिता हैं।";
    let input = "తెలుగు భాష దక్షిణ భారతదేశంలో మాతృభాషగా ప్రసిద్ధంగా ఉంది. ఈ భాష తెలుగు లిపితో రాసుకొనబడతాయి. ఇది కవిత, సాహిత్యం, మరియు కళాశిల్పం లో మంచి పరిణామం చేస్తుంది.";
    let new_one = SpecialStrClump::new(&input);
    let temp = new_one.into_iter().map(|str_| str_.to_owned()).collect::<Vec<String>>();
    assert_eq!(temp , vec!["తెలుగు" ,"భాష" ,"దక్షిణ" ,"భారతదేశంలో" ,"మాతృభాషగా" ,"ప్రసిద్ధంగా" ,"ఉంది." ,"ఈ" ,"భాష" ,"తెలుగు" ,"లిపితో" ,"రాసుకొనబడతాయి." ,"ఇది" ,"కవిత," ,"సాహిత్యం," ,"మరియు" ,"కళాశిల్పం" ,"లో" ,"మంచి" ,"పరిణామం" ,"చేస్తుంది."]);
}


#[test]
fn opening_and_tokenising() {
    let mut new_ = read_csv(r#"testing_data/_archive/training.1600000.processed.noemoticon.csv"#, true, false).unwrap();
    //let start_time = std::time::Instant::now();
    //new_.set_headers(vec!["texthash", "text" , "selected_text" , "sentiment_target" ,"time", "age of user" , "country" , "population" , "area" , "density" ]);
    // new_.describe();
    // new_.head();
    //new_.describe_the("text", false);
    //println!("Time taken to describe is : {:?}", start_time.elapsed());
    let mut temp = Tokens::new(new_.number_of_samples as usize);
    let start_time = std::time::Instant::now();
    temp.tokenise(&new_, 5 , 2 , Some(vec![' ', ',' , '.' , '!' , '(' , ')']));
    //dbg!(&temp.data_in_sequence[0..=5]);
    //temp.remove_sparse_tokens(5);
    //temp.remove_sparse_tokens(3);
    //temp.stemm_tokens(vec!["e"], true, true);
    println!("Before stemming 'run' occurs : {} times, and 'running occurs : {} times' ", temp.get_count("run"), temp.get_count("running"));
    //temp.stemm_tokens(vec![], false, false);
    println!("After stemming 'run' occurs : {} times, and 'running occurs : {} times' ", temp.get_count("run"), temp.get_count("running"));
    println!("Total number of strings tokenised : {}", new_.number_of_samples);
    println!("Total number of unique tokens : {}", temp.token_map_index.len());
    println!("");
    println!("Time taken to tokenise : {:?}", start_time.elapsed());
}

#[test]
fn big_file_tokenise_benchmark() {
    let start_time = std::time::Instant::now();

    use std::io::Write;

    let new_text = std::fs::read_to_string(r#"testing_data/shake.txt"#).expect("this file does not even exist man!!");
    //vec![' ' , '.' , '\u{feff}' , '\n' , '*' , '\r' , ',']
    let special = SpeciaStrDivideCustom::new(&new_text , vec![' ' , '.' , '\u{feff}' , '\n' , '*' , '\r' , ',' , '?' , '-', '’', '"', '`']);

    let tokkk = special.into_iter().collect::<Vec<&str>>();

    println!("number of individual tokens is : {}" ,tokkk.len());

    println!("Time taken to tokenise and vectorise is : {:?}", start_time.elapsed());

    let mut new_file = std::fs::File::create(r#"testing_data/bake.txt"#).expect("couldn't create a new file");
    let strin = format!("{:?}", tokkk);
    let start_time = std::time::Instant::now();

    new_file.write_all(strin.as_bytes());

    println!("Time taken to write is : {:?}", start_time.elapsed());
    
}

#[test]
fn stemm_individual() {

    //let file = std::fs::read_to_string(r#"testing_data/gist_stopwords.txt"#).unwrap();

    //let stop_words = file.split_ascii_whitespace().map(|s| s.to_owned()).collect();
    let stop_words = vec!["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"];
    //print!("{:?}",stop_words);

    while(true) {
        let mut hava = String::new();
        std::io::stdin().read_line(&mut hava);
        let mut the_vec = stemm_string(&hava, "clump_special", vec!["y" , "e"]);
        println!("Stemmed string is : {:?}", the_vec);
        remove_stop_words(&mut the_vec, &stop_words);
        println!("Stemmed and removed string is : {:?}", the_vec);
    }

}

#[test] 

fn opening_tokenising_and_traingin() {
    let start_time = std::time::Instant::now();



    println!("Task completed in : {:?}" , start_time.elapsed());
}