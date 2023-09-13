use std::time;

use crate::{supervised::naive_bayes::gaussian_NB::{*}, file_handling::read_from::read_csv, evaluation::accuracy::accuracy_score};


#[test]
fn test_sample() {
    let start_time = time::Instant::now();
    let mut data = read_csv("C:/Users/HARSHA/Downloads/IRIS.csv", true , false).unwrap(); 
    let h = data.train_test_split(0.5, data.get_target_index("species").unwrap() , true);
    let mut hava = gaussian_NB();
    hava.fit(&h.0, &h.1);
    dbg!(accuracy_score(&hava, &h.2, &h.3));
    print!("{:?}", start_time.elapsed());
}