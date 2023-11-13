use crate::file_handling::read_from::read_csv;
#[cfg(test)]
use crate::neural_networks::neural_network::set_leaky_value;
use super::neural_network::{ActivationFunction, NeuralNet};




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
    let mut temp = read_csv(r#"testing_data/Iris.csv"#, true, false).unwrap();
    temp.remove_columns(&vec![0]);
    let hava = NeuralNet::new(&temp, vec![4], vec![4, 5], vec![ActivationFunction::ReLu, ActivationFunction::ReLu], super::neural_network::CostFunction::BCE, 0.1, 4);
    hava.debug_biases();
    hava.debug_weights();
    println!("{:?}", hava.get_layer_detes());
}