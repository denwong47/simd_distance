use std::ops::{Add, Mul, Sub};
use std::simd::{prelude::*, LaneCount, SimdElement, StdFloat, SupportedLaneCount};

use num_traits::Float;
use rayon::prelude::*;

pub fn cartesian_elementwise<'a, T>(lhs_x: T, lhs_y: T, rhs_x: T, rhs_y: T) -> T
where
    T: Float + Copy,
{
    let diff_x = rhs_x - lhs_x;
    let diff_y = rhs_y - lhs_y;
    (diff_x * diff_x + diff_y * diff_y).sqrt()
}

pub fn cartesian_simd<T, const N: usize>(
    lhs_x: Simd<T, N>,
    lhs_y: Simd<T, N>,
    rhs_x: Simd<T, N>,
    rhs_y: Simd<T, N>,
) -> Simd<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>:
        StdFloat + Add<Output = Simd<T, N>> + Sub<Output = Simd<T, N>> + Mul<Output = Simd<T, N>>,
{
    let diff_x = rhs_x - lhs_x;
    let diff_y = rhs_y - lhs_y;
    (diff_x.mul_add(diff_x, diff_y * diff_y)).sqrt()
}

pub fn cartesian_seq_simd<T>(
    lhs_x: &[T],
    lhs_y: &[T],
    rhs_x: &[T],
    rhs_y: &[T],
) -> Vec<T>
where
    T: SimdElement,
    Simd<T, 64>: StdFloat
        + Add<Output = Simd<T, 64>>
        + Sub<Output = Simd<T, 64>>
        + Mul<Output = Simd<T, 64>>,
{
    static CHUNKS_LENGTH: usize = 64;

    lhs_x.chunks_exact(CHUNKS_LENGTH)
    .zip(lhs_y.chunks_exact(CHUNKS_LENGTH))
    .zip(rhs_x.chunks_exact(CHUNKS_LENGTH))
    .zip(rhs_y.chunks_exact(CHUNKS_LENGTH))
    .fold(
        Vec::with_capacity(lhs_x.len()),
        |mut v, (((lhs_x, lhs_y), rhs_x), rhs_y)| {
            let lhs_x_simd = Simd::<T, 64>::from_slice(lhs_x);
            let lhs_y_simd = Simd::<T, 64>::from_slice(lhs_y);
            let rhs_x_simd = Simd::<T, 64>::from_slice(rhs_x);
            let rhs_y_simd = Simd::<T, 64>::from_slice(rhs_y);

            let result = cartesian_simd::<T, 64>(lhs_x_simd, lhs_y_simd, rhs_x_simd, rhs_y_simd);

            v.extend(result.as_array());
            v
        }
    )
}

pub fn cartesian_par_simd<T>(
    lhs_x: &[T],
    lhs_y: &[T],
    rhs_x: &[T],
    rhs_y: &[T],
) -> Vec<T>
where
    T: SimdElement + Sync + Send,
    Simd<T, 64>: StdFloat
        + Add<Output = Simd<T, 64>>
        + Sub<Output = Simd<T, 64>>
        + Mul<Output = Simd<T, 64>>,
    [T]: ParallelSlice<T>,
{
    static CHUNKS_LENGTH: usize = 64;

    lhs_x
        .par_chunks_exact(CHUNKS_LENGTH)
        .zip(lhs_y.par_chunks_exact(CHUNKS_LENGTH))
        .zip(rhs_x.par_chunks_exact(CHUNKS_LENGTH))
        .zip(rhs_y.par_chunks_exact(CHUNKS_LENGTH))
        .map(|(((lhs_x, lhs_y), rhs_x), rhs_y)| {
            let lhs_x_simd = Simd::<T, 64>::from_slice(lhs_x);
            let lhs_y_simd = Simd::<T, 64>::from_slice(lhs_y);
            let rhs_x_simd = Simd::<T, 64>::from_slice(rhs_x);
            let rhs_y_simd = Simd::<T, 64>::from_slice(rhs_y);

            let result = cartesian_simd::<T, 64>(lhs_x_simd, lhs_y_simd, rhs_x_simd, rhs_y_simd);

            Vec::from(result.as_array())
        })
        .reduce(
            || Vec::<T>::with_capacity(CHUNKS_LENGTH),
            |mut v1, v2| {
                v1.extend(v2);
                v1
            },
        )
}

pub fn cartesian_par_elementwise<T>(
    lhs_x: &[T],
    lhs_y: &[T],
    rhs_x: &[T],
    rhs_y: &[T],
) -> Vec<T>
where
    T: Float + Sync + Send,
{
    static CHUNKS_LENGTH: usize = 128000;

    lhs_x
        .par_chunks_exact(CHUNKS_LENGTH)
        .zip(lhs_y.par_chunks_exact(CHUNKS_LENGTH))
        .zip(rhs_x.par_chunks_exact(CHUNKS_LENGTH))
        .zip(rhs_y.par_chunks_exact(CHUNKS_LENGTH))
        .map(|(((lhs_x, lhs_y), rhs_x), rhs_y)| {
            lhs_x
                .iter()
                .zip(lhs_y.iter())
                .zip(rhs_x.iter())
                .zip(rhs_y.iter())
                .map(|(((lhs_x, lhs_y), rhs_x), rhs_y)| {
                    cartesian_elementwise(*lhs_x, *lhs_y, *rhs_x, *rhs_y)
                })
                .collect::<Vec<T>>()
        })
        .reduce(
            || Vec::<T>::with_capacity(CHUNKS_LENGTH),
            |mut v1, v2| {
                v1.extend(v2);
                v1
            },
        )
}

pub fn cartesian_par_batch_simd<T>(
    lhs_x: &[T],
    lhs_y: &[T],
    rhs_x: &[T],
    rhs_y: &[T],
) -> Vec<T>
where
    T: SimdElement + Sync + Send,
    Simd<T, 64>: StdFloat
        + Add<Output = Simd<T, 64>>
        + Sub<Output = Simd<T, 64>>
        + Mul<Output = Simd<T, 64>>,
{
    static CHUNKS_LENGTH: usize = 128000;

    lhs_x
        .par_chunks_exact(CHUNKS_LENGTH)
        .zip(lhs_y.par_chunks_exact(CHUNKS_LENGTH))
        .zip(rhs_x.par_chunks_exact(CHUNKS_LENGTH))
        .zip(rhs_y.par_chunks_exact(CHUNKS_LENGTH))
        .map(|(((lhs_x, lhs_y), rhs_x), rhs_y)| {
            const SIMD_LENGTH: usize = 64;
            lhs_x
                .chunks_exact(SIMD_LENGTH)
                .zip(lhs_y.chunks_exact(SIMD_LENGTH))
                .zip(rhs_x.chunks_exact(SIMD_LENGTH))
                .zip(rhs_y.chunks_exact(SIMD_LENGTH))
                .map(|(((lhs_x, lhs_y), rhs_x), rhs_y)| {
                    let lhs_x_simd = Simd::<T, SIMD_LENGTH>::from_slice(lhs_x);
                    let lhs_y_simd = Simd::<T, SIMD_LENGTH>::from_slice(lhs_y);
                    let rhs_x_simd = Simd::<T, SIMD_LENGTH>::from_slice(rhs_x);
                    let rhs_y_simd = Simd::<T, SIMD_LENGTH>::from_slice(rhs_y);

                    cartesian_simd(lhs_x_simd, lhs_y_simd, rhs_x_simd, rhs_y_simd)
                })
                .fold(
                    Vec::with_capacity(lhs_x.len()),
                    |mut v, result| {
                        v.extend(result.as_array());
                        v
                    },
                )
        })
        .reduce(
            || Vec::<T>::with_capacity(CHUNKS_LENGTH),
            |mut v1, v2| {
                v1.extend(v2);
                v1
            },
        )
}