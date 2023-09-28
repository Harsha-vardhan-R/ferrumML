#[cfg(test)]

#[test]

fn divide_n_print() {
    use super::tokenisation::SpecialStr;

    let input = "**Tha&khva a#% vhbavbevb{}pp'llop'.";
    let new_one = SpecialStr::new(&input);

    let temp: Vec<&str> = new_one.into_iter().collect();

    assert_eq!(temp , vec!["Tha", "khva", "a", "vhbavbevb", "pp", "llop"]);
}