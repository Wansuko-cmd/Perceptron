use wgpu::Device;

use crate::kernels::{
    elementwise::{compare::Compare, generator::Generator, math::Math, operation::Operation},
    index::index::Index,
    linalg::mat_mul::MatMul,
    reduction::Reduction,
    shape::shape::Shape,
};

pub mod elementwise;
pub mod index;
pub mod linalg;
pub mod reduction;
pub mod shape;
pub mod task;

pub struct Kernels {
    pub reduction: Reduction,
    pub compare: Compare,
    pub generator: Generator,
    pub index: Index,
    pub mat_mul: MatMul,
    pub math: Math,
    pub operation: Operation,
    pub shape: Shape,
}

impl Kernels {
    pub fn new(device: &Device) -> Self {
        Kernels {
            reduction: Reduction::new(device),
            compare: Compare::new(device),
            generator: Generator::new(device),
            index: Index::new(device),
            mat_mul: MatMul::new(device),
            math: Math::new(device),
            operation: Operation::new(device),
            shape: Shape::new(device),
        }
    }
}
