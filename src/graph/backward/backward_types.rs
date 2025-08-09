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
    LogBackward,
    ExpBackward,
    TanhBackward,
    MinBackward,
    MaxBackward,
    MeanBackward,
    SumBackward,
    MatmulBackward,
    BroadcastBackward,
    UnsqueezeBackward,
    SqueezeBackward,
    TransposeBackward,
    ReshapeBackward,
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
            BackwardType::LogBackward => write!(f, "LogBackward"),
            BackwardType::ExpBackward => write!(f, "ExpBackward"),
            BackwardType::TanhBackward => write!(f, "TanhBackward"),
            BackwardType::MinBackward => write!(f, "MinBackward"),
            BackwardType::MaxBackward => write!(f, "MaxBackward"),
            BackwardType::MeanBackward => write!(f, "MeanBackward"),
            BackwardType::SumBackward => write!(f, "SumBackward"),
            BackwardType::MatmulBackward => write!(f, "MatmulBackward"),
            BackwardType::BroadcastBackward => write!(f, "BroadcastBackward"),
            BackwardType::UnsqueezeBackward => write!(f, "UnsqueezeBackward"),
            BackwardType::SqueezeBackward => write!(f, "SqueezeBackward"),
            BackwardType::TransposeBackward => write!(f, "TransposeBackward"),
            BackwardType::ReshapeBackward => write!(f, "ReshapeBackward"),
        }
    }
}
