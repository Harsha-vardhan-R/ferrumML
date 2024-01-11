#[cfg(test)]

#[test]
fn conv_test() {
    use super::convolution::ConvolutionKernelBuilder;

    let kernel_ = ConvolutionKernelBuilder::VerticalSobel.build_with_dimen(5);
    dbg!(kernel_);

}