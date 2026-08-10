use crate::resource::buffer::CPUBuffer;

pub fn where_d0_to_d0(condition: &CPUBuffer, x: f32, y: f32, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    for (i, value) in result.iter_mut().enumerate() {
        // if文と同じ(分岐を避けることでベクトル化を狙う)
        let mask = (condition[i] > 0f32) as u32 as f32;
        *value = mask * x + (1f32 - mask) * y;
    }
}

pub fn where_d0_to_d1(condition: &CPUBuffer, x: f32, y: &CPUBuffer, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    assert_eq!(y.count(), result.count());
    for (i, value) in result.iter_mut().enumerate() {
        let mask = (condition[i] > 0f32) as u32 as f32;
        *value = mask * x + (1f32 - mask) * y[i];
    }
}

pub fn where_d1_to_d0(condition: &CPUBuffer, x: &CPUBuffer, y: f32, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    assert_eq!(x.count(), result.count());
    for (i, value) in result.iter_mut().enumerate() {
        let mask = (condition[i] > 0f32) as u32 as f32;
        *value = mask * x[i] + (1f32 - mask) * y;
    }
}

pub fn where_d1_to_d1(condition: &CPUBuffer, x: &CPUBuffer, y: &CPUBuffer, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    assert_eq!(x.count(), result.count());
    assert_eq!(y.count(), result.count());
    for (i, value) in result.iter_mut().enumerate() {
        let mask = (condition[i] > 0f32) as u32 as f32;
        *value = mask * x[i] + (1f32 - mask) * y[i];
    }
}
