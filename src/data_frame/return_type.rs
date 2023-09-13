

#[derive(Debug)]
//this is used while returning from a predict function.
pub enum return_type {
    Strings(String),
    Floats(f32),
    Category(u8),
}


impl PartialEq for return_type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (return_type::Strings(s1), return_type::Strings(s2)) => s1 == s2,
            (return_type::Floats(f1), return_type::Floats(f2)) => f1 == f2,
            (return_type::Category(c1), return_type::Category(c2)) => c1 == c2,
            _ => false, // Handle other cases or mismatched variants
        }
    }
}