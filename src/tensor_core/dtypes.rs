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

pub trait DTComp {
    fn dtype() -> DTypes;
}

impl DTComp for bool {
    fn dtype() -> DTypes {
        return DTypes::Bool;
    }
}

impl DTComp for f32 {
    fn dtype() -> DTypes {
        return DTypes::Float32;
    }
}

impl DTComp for f64 {
    fn dtype() -> DTypes {
        return DTypes::Float64;
    }
}

impl DTComp for i8 {
    fn dtype() -> DTypes {
        return DTypes::Int8;
    }
}

impl DTComp for i16 {
    fn dtype() -> DTypes {
        return DTypes::Int16;
    }
}

impl DTComp for i32 {
    fn dtype() -> DTypes {
        return DTypes::Int32;
    }
}

impl DTComp for i64 {
    fn dtype() -> DTypes {
        return DTypes::Int64;
    }
}

impl DTComp for i128 {
    fn dtype() -> DTypes {
        return DTypes::Int128;
    }
}

impl DTComp for isize {
    fn dtype() -> DTypes {
        return DTypes::Isize;
    }
}

impl DTComp for u8 {
    fn dtype() -> DTypes {
        return DTypes::Uint8;
    }
}

impl DTComp for u16 {
    fn dtype() -> DTypes {
        return DTypes::Uint16;
    }
}

impl DTComp for u32 {
    fn dtype() -> DTypes {
        return DTypes::Uint32;
    }
}

impl DTComp for u64 {
    fn dtype() -> DTypes {
        return DTypes::Uint64;
    }
}

impl DTComp for u128 {
    fn dtype() -> DTypes {
        return DTypes::Uint128;
    }
}

impl DTComp for usize {
    fn dtype() -> DTypes {
        return DTypes::Usize;
    }
}

impl std::fmt::Display for DTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DTypes::Bool => write!(f, "bool"),
            DTypes::Float32 => write!(f, "float32"),
            DTypes::Float64 => write!(f, "float64"),
            DTypes::Int8 => write!(f, "int8"),
            DTypes::Int16 => write!(f, "int16"),
            DTypes::Int32 => write!(f, "int32"),
            DTypes::Int64 => write!(f, "int64"),
            DTypes::Int128 => write!(f, "int128"),
            DTypes::Isize => write!(f, "isize"),
            DTypes::Uint8 => write!(f, "uint8"),
            DTypes::Uint16 => write!(f, "uint16"),
            DTypes::Uint32 => write!(f, "uint32"),
            DTypes::Uint64 => write!(f, "uint64"),
            DTypes::Uint128 => write!(f, "uint128"),
            DTypes::Usize => write!(f, "usize"),
        }
    }
}
