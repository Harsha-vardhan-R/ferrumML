#![allow(non_camel_case_types)]
#![allow(non_snake_case, non_camel_case_types, unused_mut, unused_imports)]
use std::{time, mem};
use crate::file_handling::read_from::read_csv;

#[cfg(test)]

#[test]
fn opening_df() {
    let start_time = time::Instant::now();
    let mut data = read_csv("C:/Users/HARSHA/Downloads/mnist_train.csv/mnist_train.csv", true , true).unwrap();
    //data.print_headers();
    print!("{:?}", start_time.elapsed());
    data.describe_the("7x3", true);
    //data.null_stats();
    //data.print_headers();
    print!("{:?}", start_time.elapsed());
}


#[test]
fn opening_df_2() {
    let start_time = time::Instant::now();
    let mut data = read_csv("testing_data/wine-clustering.csv", true , false).unwrap();
    //data.print_headers();
    //data.head();
    data.head();
    data.normalize();
    //data.head();
    data.describe();
    data.null_stats();
    print!("{:?}", start_time.elapsed());
}


#[test]
fn opening_df_3() {
    let start_time = time::Instant::now();
    let mut data = read_csv("testing_data/archive (1)/ds_salaries.csv", true , false).unwrap();
    //data.print_headers();
    //data.head();
    data.describe();    
    //data.describe_the("salary_in_usd", true);
    //data.normalize();
    //data.describe_the("salary_in_usd", true);
    dbg!(data.get_shape());
    //data.head();
    data.remove_columns(&vec![0]);
    data.encode("experience_level");
    //data.encode("employment_type");
    data.encode("job_title");
    data.encode("salary_currency");
    //data.encode("employee_residence");
    data.encode("company_location");
    data.encode("company_size");

    data.normalize();

    data.describe();

    //let h = data.train_test_split(0.2, 10, true);
    
    println!("{:?}", start_time.elapsed());

}


