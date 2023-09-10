#![allow(non_snake_case)]
#![allow(warnings)]


pub mod n_dimen;

pub mod preprocessing {
    pub mod pca {
        pub mod pca;
        pub mod pca_test;
    }
}


pub mod clustering {
    pub mod k_means_clustering {
        mod k_means_test;
        pub mod k_means_clustering;        
    }
    pub mod heiarchial_clustering {
        pub mod heiarchial_clustering;
        mod heiarchical_clustering_tests;
    }
    pub mod dbscan {
        pub mod dbscan;
        mod dbscan_test;
    }
}

pub mod supervised {
    pub mod naive_bayes {
        pub mod naive_bayes;
        mod naive_bayes_test;
    }
    pub mod decision_trees {
        pub mod decision_trees;
        mod decision_trees_test;
    }
    pub mod linear_regression {
        pub mod linear_regression;
        mod linear_regression_test;
    }
    pub mod logistic_regression{
        pub mod logistic_regression;
        mod logistic_regression_test;
    }
    pub mod support_vector_machines {
        pub mod support_vector_machines;
        mod support_vector_machines_test;
    }
    pub mod random_forest {
        pub mod random_forest;
        mod random_forest_test;
    }
    pub mod gradient_boosting_machines {
        pub mod gradient_boosting_machines;
        mod gradient_boosting_machines_test;
    }
    pub mod k_nearest_neighbours {
        pub mod k_nearest_neighbours;
        mod k_nearest_neighbours_test;
    }
    pub mod multi_layer_perceptron {
        pub mod multi_layer_perceptron;
        mod multi_layer_perceptron_test;
    }

}
    

pub mod csv_handling;
mod csv_handling_test;

pub mod general;


fn main() {
    println!("Hello world!");

    println!("Now fuck off");
    
    println!("i am so sicking tired of this shit literally feels like i have some kind of constraint or something");
}
