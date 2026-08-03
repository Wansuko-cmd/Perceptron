use crate::resource::buffer::CPUBuffer;

pub fn where_d0_to_d0(condition: &CPUBuffer, x: f32, y: f32, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    for (i, value) in result.value.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x } else { y };
    }
}

pub fn where_d0_to_d1(condition: &CPUBuffer, x: f32, y: &CPUBuffer, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    assert_eq!(y.count(), result.count());
    for (i, value) in result.value.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x } else { y[i] };
    }
}

pub fn where_d1_to_d0(condition: &CPUBuffer, x: &CPUBuffer, y: f32, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    assert_eq!(x.count(), result.count());
    for (i, value) in result.value.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x[i] } else { y };
    }
}

pub fn where_d1_to_d1(condition: &CPUBuffer, x: &CPUBuffer, y: &CPUBuffer, result: &mut CPUBuffer) {
    assert_eq!(condition.count(), result.count());
    assert_eq!(x.count(), result.count());
    assert_eq!(y.count(), result.count());
    for (i, value) in result.value.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x[i] } else { y[i] };
    }
}
