use std::ops::{Index, IndexMut};

pub struct CPUBuffer {
    pub value: Vec<f32>,
}

impl CPUBuffer {
    pub fn create(size: usize) -> CPUBuffer {
        let value = vec![0f32; size];
        CPUBuffer { value: value }
    }

    pub fn init(value: &[f32]) -> CPUBuffer {
        CPUBuffer { value: value.to_vec() }
    }

    pub fn write(&mut self, index: usize, value: f32) {
        self.value[index] = value
    }

    pub fn count(&self) -> usize {
        self.value.len()
    }
}

impl Index<usize> for CPUBuffer {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.value[index]
    }
}

impl IndexMut<usize> for CPUBuffer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.value[index]
    }
}

impl PartialEq for CPUBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
