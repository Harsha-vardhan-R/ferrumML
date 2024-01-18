use super::{ neural_network::{NeuralNet, functionValueAt}, convolution_architecture::ConvolutionArchitecture};

///! This file contains the implementation for the type `Pipe`
///! The `Pipe` is the object connecting the `ConvolutionArchitecture` and `NeuralNet`
///! It takes ownership of both of those structs.



struct Pipe<U : functionValueAt, T> {
    architecture : ConvolutionArchitecture<U>,
    network : NeuralNet<T>
}