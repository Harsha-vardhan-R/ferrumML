

#[derive(Debug)]
//this is used while returning from a predict function.
pub enum ReturnType {
    Strings(String),
    Floats(f32),
    Category(u8),
}


impl PartialEq for ReturnType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ReturnType::Strings(s1), ReturnType::Strings(s2)) => s1 == s2,
            (ReturnType::Floats(f1), ReturnType::Floats(f2)) => f1 == f2,
            (ReturnType::Category(c1), ReturnType::Category(c2)) => c1 == c2,
            _ => false, // Handle other cases or mismatched variants
        }
    }
}