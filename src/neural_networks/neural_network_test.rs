#[cfg(test)]
use crate::neural_networks::neural_network::set_leaky_value;
use super::neural_network::ActivationFunction;




#[test]
fn activation_function() {
    

    let temp = ActivationFunction::Sigmoid;
    //set_leaky_value(0.5);

    for i in -10..10 {
        println!("{:?} | {:?}", temp.activation_function_at(i as f32) , temp.derivative_at(i as f32));
    }
}