#[derive(Debug, Copy, Clone)]
pub enum DTypes {
    Bool,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Isize,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,
    Usize,
}

pub trait DTypeMarker {
    fn dtype() -> DTypes;
}

impl DTypeMarker for bool {
    fn dtype() -> DTypes {
        return DTypes::Bool;
    }
}

impl DTypeMarker for f32 {
    fn dtype() -> DTypes {
        return DTypes::Float32;
    }
}

impl DTypeMarker for f64 {
    fn dtype() -> DTypes {
        return DTypes::Float64;
    }
}

impl DTypeMarker for i8 {
    fn dtype() -> DTypes {
        return DTypes::Int8;
    }
}

impl DTypeMarker for i16 {
    fn dtype() -> DTypes {
        return DTypes::Int16;
    }
}

impl DTypeMarker for i32 {
    fn dtype() -> DTypes {
        return DTypes::Int32;
    }
}

impl DTypeMarker for i64 {
    fn dtype() -> DTypes {
        return DTypes::Int64;
    }
}

impl DTypeMarker for i128 {
    fn dtype() -> DTypes {
        return DTypes::Int128;
    }
}

impl DTypeMarker for isize {
    fn dtype() -> DTypes {
        return DTypes::Isize;
    }
}

impl DTypeMarker for u8 {
    fn dtype() -> DTypes {
        return DTypes::Uint8;
    }
}

impl DTypeMarker for u16 {
    fn dtype() -> DTypes {
        return DTypes::Uint16;
    }
}

impl DTypeMarker for u32 {
    fn dtype() -> DTypes {
        return DTypes::Uint32;
    }
}

impl DTypeMarker for u64 {
    fn dtype() -> DTypes {
        return DTypes::Uint64;
    }
}

impl DTypeMarker for u128 {
    fn dtype() -> DTypes {
        return DTypes::Uint128;
    }
}

impl DTypeMarker for usize {
    fn dtype() -> DTypes {
        return DTypes::Usize;
    }
}
