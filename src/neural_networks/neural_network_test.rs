

use std::process::exit;

use crate::data_frame::{data_frame::*, data_type::DataType};
use rand::random;
use sprs::DenseVector;

use crate::{file_handling::read_from::read_csv, neural_networks::neural_network::OutputMap, trait_definition::MLalgo, data_frame::data_type::print_at_index};
#[cfg(test)]
use crate::neural_networks::neural_network::set_leaky_value;
use super::neural_network::{ActivationFunction, NeuralNet, CostFunction};
use plotters::prelude::*;



#[test]
fn activation_function() {
    let temp = ActivationFunction::Sigmoid;
    //set_leaky_value(0.5);

    // for i in -10..10 {
    //     println!("{:?} | {:?}", temp.activation_function_at(i as f32) , temp.derivative_at(i as f32));
    // }
}



#[test]
fn new_Struct() {
    let mut temp = read_csv(r#"testing_data/concrete.csv"#, true, false).unwrap();
    temp.remove_columns(&vec![2, 4]);
    temp.normalize();
    temp.describe();
    let activation_function = vec![ActivationFunction::ReLu,  ActivationFunction::ReLu , ActivationFunction::ReLu];
    let mut neural_net = NeuralNet::new(&temp, vec![6], vec![4, 4], activation_function, super::neural_network::CostFunction::MSE, -0.001, OutputMap::SoftMax, 6);
    
    // let mut hava = neural_net;
    
    // println!("{:?}", hava.feed_forward(&vec![5.1,3.5]));
    // hava.debug_biases();
    // hava.debug_weights();
    // println!("{:?}", hava.get_layer_detes());
    // hava.debug_activation_values();
    let (X_train, y_train, X_test, y_test) = temp.train_test_split(0.8, 6, false);
    neural_net.fit(&X_train, &y_train);

    // for (index, value) in X_test.iter().enumerate() {
    //     print!("predicted value : {:?}, actual value : ", neural_net.predict_float(value));
    //     y_test.print_at(index);
    //     println!();
    // }
}

#[test]
fn sin_validation() -> Result<(), Box<dyn std::error::Error>> {
    let mut train_x = vec![0.0_f32; 1000];
    let mut train_y = vec![0.0_f32; 1000];
    for i in 0..1000 {
        train_x[i] = (random::<f32>())*8.0;
        train_y[i] = f32::sin(train_x[i]);
    }

    let mut df  = DataFrame::new();
    df.new_column(DataType::Floats(train_x.clone()), 0);
    df.new_column(DataType::Floats(train_y.clone()), 1);
    df.describe();
    df.head();

    let (X_train, y_train, X_test, y_test ) = df.train_test_split(0.0, 1, false);
    let yyyy = match y_train {
        DataType::Floats(ref temp) => temp,
        _ => panic!("no wayyyyy!"),
    };

    

    let mut neural_net = NeuralNet::new(&df, vec![1], vec![64],
        vec![ActivationFunction::Tanh ,ActivationFunction::Tanh], CostFunction::MSE,
        -0.001, OutputMap::ArgMax, 1);

    neural_net.set_bias_clip_value(1.0);
    neural_net.set_weight_clip_value(1.0);
    neural_net.xavier_weights();
    //neural_net.debug_weights();

    

    // neural_net.debug_activation_values();
    // neural_net.debug_biases();
    // neural_net.debug_chained_derivaives();
    // neural_net.debug_net_values();

    let mut initinit = 0;

    for i in 1..100 {
        for (index, frame) in X_train.iter().enumerate() {
            neural_net.feed_forward_back_propogate(frame, &vec![yyyy[index]]);

            if (index%100 == 0) {
                let filename = format!("dummy/scatter{:04}.png", initinit);
                let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
                root.fill(&WHITE)?;
                let mut chart = ChartBuilder::on(&root)
                .caption("Scatter Plot", ("Arial", 20))
                .x_label_area_size(40)
                .y_label_area_size(40)
                .build_cartesian_2d(0.0..9.0, -1.0..1.0)?;

                chart.draw_series(
                    train_x.iter().zip(train_y.iter()).map(|(x, y)| {
                        Circle::new((*x as f64, *y as f64), 2, ShapeStyle::from(&BLACK).filled())
                    }),
                ).map_err(|e| e.to_string());

                chart.draw_series(
                    train_x.iter().zip(train_y.iter()).map(|(x, y)| {
                        Circle::new((*x as f64, neural_net.predict_float(&vec![*x]) as f64), 1, ShapeStyle::from(&BLUE))
                    }),
                ).map_err(|e| e.to_string());initinit += 1;
            }
            
        }
    }
    
    
    Ok(())
}

