use std::time;

use crate::{csv_handling::{data_frame, read_csv}, supervised::naive_bayes::naive_bayes::{gaussian_NB , *}, general::accuracy_score};


#[test]

fn test_sample() {

    let start_time = time::Instant::now();

    let mut data = read_csv("C:/Users/HARSHA/Downloads/IRIS.csv", true , false).unwrap();


    //data.null_stats();

    //data.describe();

    //data.head();

    //data.normalize();
 
    let h = data.train_test_split(0.5, data.get_target_index("species").unwrap() , true);

    let mut hava = gaussian_NB();

    hava.fit(&h.0, &h.1); 
    //hava.get_gaussian_vector();

    dbg!(accuracy_score(&hava, &h.2, &h.3));

    //dbg!(&hava);

    print!("{:?}", start_time.elapsed());

}