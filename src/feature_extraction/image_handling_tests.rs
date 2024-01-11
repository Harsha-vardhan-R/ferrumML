#[cfg(test)]


#[test]
fn image_split_and_save() {
    use image::io::Reader as ImageReader;

    let file_path = r#"/home/harshavardhan/Desktop/tux.png"#;

    let image = ImageReader::open(file_path).unwrap().decode().unwrap();
    
    print!("{:?}", image.as_flat_samples_u8());
}