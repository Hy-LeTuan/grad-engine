use std::fmt::Debug;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphJSON<T> {
    pub root: NodeJSON<T>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeJSON<T> {
    pub name: String,
    pub origin: TensorJSON<T>,
    pub gradient: TensorJSON<T>,
    pub children: Vec<NodeJSON<T>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorJSON<T> {
    pub data: Vec<T>,
    pub offset: Option<usize>,
    pub shape: Vec<usize>,
}
