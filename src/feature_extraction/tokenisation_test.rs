#[cfg(test)]

#[test]

fn divide_n_print() {
    use super::tokenisation::SpecialStr;

    let input = String::from("The hungry !ass .dog.");
    let new_one = SpecialStr::new(&input);

    for i in new_one.iter() {
        println!("{}",i) ;
    }

}