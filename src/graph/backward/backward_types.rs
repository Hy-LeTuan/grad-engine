use std::fmt;

#[derive(Debug)]
pub enum BackwardType {
    GradAccum,
    AddBackward,
    SubBackward,
    MulBackward,
    DivBackward,
    DotBackward,
    LnBackward,
}

impl fmt::Display for BackwardType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BackwardType::GradAccum => write!(f, "GradAccum"),
            BackwardType::AddBackward => write!(f, "AddBackward"),
            BackwardType::SubBackward => write!(f, "SubBackward"),
            BackwardType::MulBackward => write!(f, "MulBackward"),
            BackwardType::DivBackward => write!(f, "DivBackward"),
            BackwardType::DotBackward => write!(f, "DotBackward"),
            BackwardType::LnBackward => write!(f, "LnBackward"),
        }
    }
}
