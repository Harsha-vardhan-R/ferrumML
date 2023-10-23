use std::time;

use crate::{file_handling::read_from::read_csv, evaluation::accuracy::accuracy_score, supervised::naive_bayes::gaussian_NB::GaussianNb, trait_definition::MLalgo};




#[test]
fn test_sample_gaussian() {
    let start_time = time::Instant::now();
    let mut data = read_csv(r#" "#, true , true).unwrap(); 
    print!("loading the data : {:?}", start_time.elapsed());
    //dalnta.head();
    let h = data.train_test_split(0.1, data.get_target_index("label").unwrap() , true);
    println!("splitting the data : {:?}", start_time.elapsed());
    let mut hava = GaussianNb::new();
    hava.fit(&h.0, &h.1);
    println!("fitting the data : {:?}", start_time.elapsed());
    dbg!(accuracy_score(&hava, &h.2, &h.3));
    print!("{:?}", start_time.elapsed());
}

#[test]
fn test_sample_multinomial() {
    let start_time = time::Instant::now();
    let df = read_csv("", true, false).unwrap();
    let h = df.train_test_split(0.4, 0, true);
    print!("{:?}", start_time.elapsed());   
}