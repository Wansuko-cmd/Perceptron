pub fn random_d1(from: f32, until: f32, seed: u64, result: &mut [f32]) {
    let mut state = if seed == 0 { 1 } else { seed };
    let range = until - from;
    for res in result.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = state as f32 / u64::MAX as f32;
        *res = from + unit * range;
    }
}
