#![allow(non_snake_case, warnings)]

pub mod preprocessing {
    pub mod pca;
    mod pca_test;
    
}

pub mod n_dimen {
    pub mod n_dimen;
}

pub mod evaluation {
    pub mod accuracy;
}

pub mod data_frame {
    pub mod data_frame;
    mod data_frame_test;
    pub mod data_type;
    pub mod return_type;
}

pub mod file_handling {
    pub mod read_from;
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
        pub mod gaussian_NB;
        pub mod multinomial_NB;
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

}


fn main() {
    println!("Hello world!");
    println!("Now fuck off");
	println!("Suddenly started to code this on ubuntu, I am such a loser man.");
}
