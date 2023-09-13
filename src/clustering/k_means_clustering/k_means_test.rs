use std::time;

#[cfg(test)]
use crate::clustering::k_means_clustering::k_means_clustering::k_means_df;

#[test]

fn k_means_test_wine() {
    let start_time = time::Instant::now();

    let mut data_frame = k_means_df("wine-clustering.csv", vec![]);
    //data_frame.normalize(); 
    data_frame.fit( 100, 0.001 , 3);
    data_frame.print_populations();

    print!("Time Taken: ");
    print!("{:?}\n", time::Instant::now() - start_time);
} 

#[test]

fn k_means_test_iris() {
    let start_time = time::Instant::now();

    let mut data_frame = k_means_df("IRIS.csv", vec![0,1,2,3]);
    data_frame.head();
    //data_frame.remove_columns(&vec![0,1]);
    //data_frame.head();
    data_frame.normalize();
    data_frame.head();
    data_frame.remove_columns(&vec![0,2]);
    data_frame.head();

    //data_frame.normalize(); 
    //data_frame.fit( 1000, 0.001, 3);
    //data_frame.print_populations();
    //data_frame.post_scatter_plot("image.png", 0,1);
    
    print!("Time Taken: ");
    print!("{:?}\n", time::Instant::now() - start_time);
    
}

