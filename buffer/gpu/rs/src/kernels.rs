use wgpu::Device;

use crate::kernels::{
    elementwise::{compare::Compare, math::Math, operation::Operation},
    index::index::Index,
    linalg::mat_mul::MatMul,
    reduction::collection::Collection,
    shape::shape::Shape,
};

pub mod elementwise;
pub mod index;
pub mod linalg;
pub mod reduction;
pub mod shape;
pub mod task;

pub struct Kernels {
    pub collection: Collection,
    pub compare: Compare,
    pub index: Index,
    pub mat_mul: MatMul,
    pub math: Math,
    pub operation: Operation,
    pub shape: Shape,
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
            shape: Shape::new(device),
        }
    }
}
