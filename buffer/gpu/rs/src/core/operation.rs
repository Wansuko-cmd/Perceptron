use crate::core::buffer::GPUBuffer;
use crate::core::context::Context;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

pub fn plus_with_d0_to_d1(
    x: f32,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let x = &GPUBuffer::init(&[x], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 0];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d0_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d1_to_d0(
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    let y = &GPUBuffer::init(&[y], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 0];

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d1_to_d0",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d1_to_d1(
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d1_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d1_to_d2(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, yi, yj].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, 0, yj, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d1_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d1_to_d3(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d1_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d2_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, xi, xj].map(|v| v as u32);
    let x_stride = [0, 0, xj, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d2_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d2_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match (axis1, axis2) {
        (0, 1) => [0, xj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, xj, 0, 1].map(|v| v as u32),
        _ => [0, 0, xj, 1].map(|v| v as u32),
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d2_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d3_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d3_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d3_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [0, yj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d3_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d3_to_d4(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [yi, yj, yk, yl].map(|v| v as u32);
    let x_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [xj * xk, xk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [xj * xk, xk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [xj * xk, 0, xk, 1].map(|v| v as u32),
        _ => [0, xj * xk, xk, 1].map(|v| v as u32),
    };
    let y_stride = [yj * yk * yl, yk * yl, yl, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d3_to_d4",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d4_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [1, 0, 0, 0],
        1 => [0, 1, 0, 0],
        2 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d4_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d4_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [yj, 1, 0, 0].map(|v| v as u32),
        (0, 2) => [yj, 0, 1, 0].map(|v| v as u32),
        (0, 3) => [yj, 0, 0, 1].map(|v| v as u32),
        (1, 2) => [0, yj, 1, 0].map(|v| v as u32),
        (1, 3) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d4_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn plus_with_d4_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [yj * yk, yk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [yj * yk, yk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [yj * yk, 0, yk, 1].map(|v| v as u32),
        _ => [0, yj * yk, yk, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.plus_d4,
        "plus_with_d4_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d0_to_d1(
    x: f32,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let x = &GPUBuffer::init(&[x], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 0];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d0_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d1_to_d0(
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    let y = &GPUBuffer::init(&[y], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 0];

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d1_to_d0",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d1_to_d1(
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d1_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d1_to_d2(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, yi, yj].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, 0, yj, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d1_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d1_to_d3(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d1_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d2_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, xi, xj].map(|v| v as u32);
    let x_stride = [0, 0, xj, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d2_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d2_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match (axis1, axis2) {
        (0, 1) => [0, xj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, xj, 0, 1].map(|v| v as u32),
        _ => [0, 0, xj, 1].map(|v| v as u32),
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d2_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d3_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d3_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d3_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [0, yj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d3_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d3_to_d4(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [yi, yj, yk, yl].map(|v| v as u32);
    let x_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [xj * xk, xk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [xj * xk, xk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [xj * xk, 0, xk, 1].map(|v| v as u32),
        _ => [0, xj * xk, xk, 1].map(|v| v as u32),
    };
    let y_stride = [yj * yk * yl, yk * yl, yl, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d3_to_d4",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d4_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [1, 0, 0, 0],
        1 => [0, 1, 0, 0],
        2 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d4_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d4_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [yj, 1, 0, 0].map(|v| v as u32),
        (0, 2) => [yj, 0, 1, 0].map(|v| v as u32),
        (0, 3) => [yj, 0, 0, 1].map(|v| v as u32),
        (1, 2) => [0, yj, 1, 0].map(|v| v as u32),
        (1, 3) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d4_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn minus_with_d4_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [yj * yk, yk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [yj * yk, yk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [yj * yk, 0, yk, 1].map(|v| v as u32),
        _ => [0, yj * yk, yk, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.minus_d4,
        "minus_with_d4_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d0_to_d1(
    x: f32,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let x = &GPUBuffer::init(&[x], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 0];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d0_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d1_to_d0(
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    let y = &GPUBuffer::init(&[y], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 0];

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d1_to_d0",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d1_to_d1(
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d1_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d1_to_d2(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, yi, yj].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, 0, yj, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d1_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d1_to_d3(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d1_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d2_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, xi, xj].map(|v| v as u32);
    let x_stride = [0, 0, xj, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d2_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d2_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match (axis1, axis2) {
        (0, 1) => [0, xj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, xj, 0, 1].map(|v| v as u32),
        _ => [0, 0, xj, 1].map(|v| v as u32),
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d2_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d3_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d3_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d3_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [0, yj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d3_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d3_to_d4(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [yi, yj, yk, yl].map(|v| v as u32);
    let x_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [xj * xk, xk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [xj * xk, xk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [xj * xk, 0, xk, 1].map(|v| v as u32),
        _ => [0, xj * xk, xk, 1].map(|v| v as u32),
    };
    let y_stride = [yj * yk * yl, yk * yl, yl, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d3_to_d4",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d4_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [1, 0, 0, 0],
        1 => [0, 1, 0, 0],
        2 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d4_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d4_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [yj, 1, 0, 0].map(|v| v as u32),
        (0, 2) => [yj, 0, 1, 0].map(|v| v as u32),
        (0, 3) => [yj, 0, 0, 1].map(|v| v as u32),
        (1, 2) => [0, yj, 1, 0].map(|v| v as u32),
        (1, 3) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d4_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn times_with_d4_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [yj * yk, yk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [yj * yk, yk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [yj * yk, 0, yk, 1].map(|v| v as u32),
        _ => [0, yj * yk, yk, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.times_d4,
        "times_with_d4_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d0_to_d1(
    x: f32,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let x = &GPUBuffer::init(&[x], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 0];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d0_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d1_to_d0(
    x: &GPUBuffer,
    y: f32,
    result: &GPUBuffer,
    context: &Context,
) {
    let y = &GPUBuffer::init(&[y], context);

    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 0];

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d1_to_d0",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d1_to_d1(
    x: &GPUBuffer,
    y: &GPUBuffer,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, 1, result.count() as u32];
    let x_stride = [0, 0, 0, 1];
    let y_stride = [0, 0, 0, 1];

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d1_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d1_to_d2(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, yi, yj].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, 0, yj, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d1_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d1_to_d3(
    x: &GPUBuffer,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d1_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d2_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, 1, xi, xj].map(|v| v as u32);
    let x_stride = [0, 0, xj, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d2_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d2_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, yi, yj, yk].map(|v| v as u32);
    let x_stride = match (axis1, axis2) {
        (0, 1) => [0, xj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, xj, 0, 1].map(|v| v as u32),
        _ => [0, 0, xj, 1].map(|v| v as u32),
    };
    let y_stride = [0, yj * yk, yk, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d2_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d3_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [0, 1, 0, 0],
        1 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d3_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d3_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [1, xi, xj, xk].map(|v| v as u32);
    let x_stride = [0, xj * xk, xk, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [0, yj, 1, 0].map(|v| v as u32),
        (0, 2) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d3_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d3_to_d4(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize, yl: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [yi, yj, yk, yl].map(|v| v as u32);
    let x_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [xj * xk, xk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [xj * xk, xk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [xj * xk, 0, xk, 1].map(|v| v as u32),
        _ => [0, xj * xk, xk, 1].map(|v| v as u32),
    };
    let y_stride = [yj * yk * yl, yk * yl, yl, 1].map(|v| v as u32);

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d3_to_d4",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d4_to_d1(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer,
    axis: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match axis {
        0 => [1, 0, 0, 0],
        1 => [0, 1, 0, 0],
        2 => [0, 0, 1, 0],
        _ => [0, 0, 0, 1],
    };

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d4_to_d1",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d4_to_d2(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize,
    axis1: usize, axis2: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2) {
        (0, 1) => [yj, 1, 0, 0].map(|v| v as u32),
        (0, 2) => [yj, 0, 1, 0].map(|v| v as u32),
        (0, 3) => [yj, 0, 0, 1].map(|v| v as u32),
        (1, 2) => [0, yj, 1, 0].map(|v| v as u32),
        (1, 3) => [0, yj, 0, 1].map(|v| v as u32),
        _ => [0, 0, yj, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d4_to_d2",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

pub fn div_with_d4_to_d3(
    x: &GPUBuffer, xi: usize, xj: usize, xk: usize, xl: usize,
    y: &GPUBuffer, yi: usize, yj: usize, yk: usize,
    axis1: usize, axis2: usize, axis3: usize,
    result: &GPUBuffer,
    context: &Context,
) {
    let result_shape = [xi, xj, xk, xl].map(|v| v as u32);
    let x_stride = [xj * xk * xl, xk * xl, xl, 1].map(|v| v as u32);
    let y_stride = match (axis1, axis2, axis3) {
        (0, 1, 2) => [yj * yk, yk, 1, 0].map(|v| v as u32),
        (0, 1, 3) => [yj * yk, yk, 0, 1].map(|v| v as u32),
        (0, 2, 3) => [yj * yk, 0, yk, 1].map(|v| v as u32),
        _ => [0, yj * yk, yk, 1].map(|v| v as u32),
    };

    zip_with_d4(
        &context.pipeline.operation.div_d4,
        "div_with_d4_to_d3",
        x, y,
        result_shape,
        x_stride,
        y_stride,
        result,
        context,
    );
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OperationParams {
    result_shape: [u32; 4],
    x_stride: [u32; 4],
    y_stride: [u32; 4],
    length: u32,
    _pad: [u32; 3],
}

fn zip_with_d4(
    pipeline: &wgpu::ComputePipeline, label: &str,
    x: &GPUBuffer,
    y: &GPUBuffer,
    result_shape: [u32; 4],
    x_stride: [u32; 4],
    y_stride: [u32; 4],
    result: &GPUBuffer,
    context: &Context,
) {
    let device = &context.device;
    let queue = &context.queue;

    let param = OperationParams {
        result_shape: result_shape,
        x_stride: x_stride,
        y_stride: y_stride,
        length: result.count() as u32,
        _pad: [0; 3],
    };

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&param),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &context.pipeline.operation.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: param_buffer.as_entire_binding(),
            },
        ]
    });

    let mut encoder = device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
        label: Some(label)
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(result.workgroup_count(WORKGROUP_SIZE), 1, 1);
    }

    queue.submit(Some(encoder.finish()));
}
