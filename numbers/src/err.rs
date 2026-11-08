#[derive(PartialEq, Debug)]
pub enum LinAlgError {
    IncompatibleShape,
    Singular,
}

#[derive(PartialEq, Debug)]
pub struct ValueError;
