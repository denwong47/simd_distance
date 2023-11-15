#![feature(portable_simd)]

pub mod cartesian;

#[cfg(test)]
mod tests {
    use super::*;
    use timeit::timeit_loops;

    macro_rules! expand_funcs {
        (
            $((
                $name:ident,
                $func:ident
            )),+
        ) => {
            $(
                #[test]
                fn $name() {
                    let lhs_x= vec![0.; 12800000];
                    let lhs_y = vec![1.; 12800000];
                    let rhs_x = vec![1.; 12800000];
                    let rhs_y = vec![0.; 12800000];
        
                    let time = timeit_loops!(
                        100,
                        { cartesian::$func(&lhs_x, &lhs_y, &rhs_x, &rhs_y); }
                    );
                
                    println!("Time by {}: {}", stringify!($func), time);
                }
            )*
        };
    }

    expand_funcs!(
        (cartesian_seq_simd, cartesian_seq_simd),
        (cartesian_par_elementwise, cartesian_par_elementwise),
        (cartesian_par_simd, cartesian_par_simd),
        (cartesian_par_batch_simd, cartesian_par_batch_simd)
    );
}
