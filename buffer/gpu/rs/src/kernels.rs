use wgpu::Device;

use crate::kernels::{
    elementwise::{compare::Compare, math::Math, operation::Operation},
    index::index::Index,
    linalg::{mat_mul::MatMul, transpose::Transpose},
    reduction::collection::Collection,
};

pub mod elementwise;
pub mod index;
pub mod linalg;
pub mod reduction;
pub mod task;

pub struct Kernels {
    pub collection: Collection,
    pub compare: Compare,
    pub index: Index,
    pub mat_mul: MatMul,
    pub math: Math,
    pub operation: Operation,
    pub transpose: Transpose,
}

impl Kernels {
    pub fn new(device: &Device) -> Self {
        Kernels {
            collection: Collection::new(device),
            compare: Compare::new(device),
            index: Index::new(device),
            mat_mul: MatMul::new(device),
            math: Math::new(device),
            operation: Operation::new(device),
            transpose: Transpose::new(device),
        }
    }
}
