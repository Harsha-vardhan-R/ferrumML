use std::time;

use crate::{csv_handling::{data_frame, read_csv}, supervised::naive_bayes::naive_bayes::{gaussian_NB , *}, general::accuracy_score};


#[test]

fn test_sample() {

    let start_time = time::Instant::now();

    let mut data = read_csv("C:/Users/HARSHA/Downloads/archive (3)/nba_data_processed.csv", true , false).unwrap();
    //data.print_headers();
    //data.head();

    //data.describe();

    data.remove_columns(&vec![0,1]);

    data.describe();

    data.null_stats();

    data.interpolate_all("dumbfill");

    data.null_stats();
 
    let h = data.train_test_split(0.2, 1, true);
    
    let mut hava = gaussian_NB();

    hava.fit(&h.0, &h.1); 


    dbg!(accuracy_score(hava, &h.2, &h.3)); 

    print!("{:?}", start_time.elapsed());
}