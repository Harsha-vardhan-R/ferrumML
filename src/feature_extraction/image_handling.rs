///! various common techiniques used to split channels and manipulates.
///! written mostly to be used in compliment to the `convolution`


/// Splits the channels and returns each of them as a seperate matrix.
/// divides based on the order in which inputs are provided.
pub fn split_channels(image : &Vec<Vec<(u8, u8, u8)>>) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let (height , width) = (image.len() , image[0].len());
    let mut first_channel = vec![vec![0_u8 ; width as usize] ; height as usize];
    let mut second_channel = vec![vec![0_u8 ; width as usize] ; height as usize];
    let mut third_channel = vec![vec![0_u8 ; width as usize] ; height as usize];

    for _height in 0..height/2 {
        for _width in 0..width {
            first_channel[_height][_width] = image[_height][_width].0;
            second_channel[_height][_width] = image[_height][_width].1;
            third_channel[_height][_width] = image[_height][_width].2;

            first_channel[height - _height][_width] = image[height - _height][_width].0;
            second_channel[height - _height][_width] = image[height - _height][_width].1;
            third_channel[height - _height][_width] = image[height - _height][_width].2;
        }
    }

    (first_channel, second_channel, third_channel)
}