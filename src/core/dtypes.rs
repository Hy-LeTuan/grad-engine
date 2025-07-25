#[derive(Debug, Copy, Clone)]
pub enum DTypes {
    Float32,
    Int64,
    Bool,
}

pub trait DTypeMarker {
    fn dtype() -> DTypes;
}

impl DTypeMarker for f32 {
    fn dtype() -> DTypes {
        return DTypes::Float32;
    }
}

impl DTypeMarker for i64 {
    fn dtype() -> DTypes {
        return DTypes::Int64;
    }
}

impl DTypeMarker for bool {
    fn dtype() -> DTypes {
        return DTypes::Bool;
    }
}
