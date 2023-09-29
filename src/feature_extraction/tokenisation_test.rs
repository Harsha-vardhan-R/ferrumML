#[cfg(test)]


#[test]
fn divide_n_print() {
    use super::tokenisation::SpecialStr;

    let input = "happy bday. .i.";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();
    print!("{:?}", temp);
    //assert_eq!(temp , vec![]);
}


fn divide_n_print_2() {
    use super::tokenisation::SpecialStr;

    let input = "hvcvk (vuk == gvyv'' jvv( hvktvk";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    println!("{:?}", temp);

    //assert_eq!(temp , vec!["hvcvk", "vuk", "gvyv", "jvv", "hvktvk"]);
}