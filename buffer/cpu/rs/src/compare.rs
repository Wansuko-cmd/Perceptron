pub fn greater_than_d1_to_d0(x: &[f32], y: f32, result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = if val > y { 1f32 } else { 0f32 };
    }
}

pub fn greater_than_d1_to_d1(x: &[f32], y: &[f32], result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    assert_eq!(y.len(), result.len());
    for (i, value) in result.iter_mut().enumerate() {
        *value = if x[i] > y[i] { 1f32 } else { 0f32 };
    }
}


pub fn less_than_d1_to_d0(x: &[f32], y: f32, result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    for (&val, res) in x.iter().zip(result.iter_mut()) {
        *res = if val < y { 1f32 } else { 0f32 };
    }
}

pub fn less_than_d1_to_d1(x: &[f32], y: &[f32], result: &mut [f32]) {
    assert_eq!(x.len(), result.len());
    assert_eq!(y.len(), result.len());
    for (i, value) in result.iter_mut().enumerate() {
        *value = if x[i] < y[i] { 1f32 } else { 0f32 };
    }
}

pub fn where_d0_to_d1(condition: &[f32], x: f32, y: &[f32], result: &mut [f32]) {
    assert_eq!(condition.len(), result.len());
    assert_eq!(y.len(), result.len());
    for (i, value) in result.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x } else { y[i] };
    }
}

pub fn where_d1_to_d0(condition: &[f32], x: &[f32], y: f32, result: &mut [f32]) {
    assert_eq!(condition.len(), result.len());
    assert_eq!(x.len(), result.len());
    for (i, value) in result.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x[i] } else { y };
    }
}

pub fn where_d1_to_d1(condition: &[f32], x: &[f32], y: &[f32], result: &mut [f32]) {
    assert_eq!(condition.len(), result.len());
    assert_eq!(x.len(), result.len());
    assert_eq!(y.len(), result.len());
    for (i, value) in result.iter_mut().enumerate() {
        *value = if condition[i] > 0f32 { x[i] } else { y[i] };
    }
}
