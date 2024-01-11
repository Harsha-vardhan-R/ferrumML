#[cfg(test)]

#[test]
fn conv_test() {
    use super::convolution::ConvolutionKernelBuilder;

    let kernel_ = ConvolutionKernelBuilder::UnsharpMask.build_with_dimen(3);
    dbg!(kernel_);

}