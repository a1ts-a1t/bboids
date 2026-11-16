use crate::Matrix;
use crate::arith::{Magnitude, Scalar};
use crate::matrix::Column;
use crate::matrix::ViewMut;

pub trait PivotSearch: Sized {
    fn pivot_row(column: &Column<Self>) -> Option<usize>;
}

impl<T: Magnitude> PivotSearch for T {
    fn pivot_row(column: &Column<Self>) -> Option<usize> {
        column
            .enumerate()
            .map(|(i, ele)| (i, ele.abs()))
            .reduce(|acc, (i, magnitude)| {
                if magnitude > acc.1 {
                    (i, magnitude)
                } else {
                    acc
                }
            })
            .map(|(i, _)| i)
    }
}

#[derive(Debug, PartialEq)]
pub enum LuInfo {
    Ok,
    ZeroValue(usize),
}

/// returns the pivots if successful
/// copying off of dgetrf2
/// https://netlib.org/lapack/explore-html/d8/d2f/group__getrf2_gaebbbe6a533dfea4454123e0553bd3575.html
fn lu_driver<T: Scalar + PivotSearch + Clone>(a: &mut ViewMut<T>) -> (Vec<usize>, LuInfo) {
    let (m, n) = a.shape();

    if m == 0 || n == 0 {
        return (Vec::new(), LuInfo::Ok);
    }

    if m == 1 {
        let info = if a[(0, 0)] == T::zero() {
            LuInfo::ZeroValue(0)
        } else {
            LuInfo::Ok
        };
        return ([0].to_vec(), info);
    }

    if n == 1 {
        let argmax_index = T::pivot_row(&a.column(0)).unwrap();
        let ipiv = [argmax_index].to_vec();

        if a[(argmax_index, 0)] == T::zero() {
            return (ipiv, LuInfo::ZeroValue(0));
        }

        if argmax_index != 0 {
            a.swap((0, 0), (argmax_index, 0));
        }

        let max = a[(0, 0)].clone();
        for i in 1..m {
            // max will always be non zero. this is the condition checked above
            a[(i, 0)] = max.inverse().unwrap() * a[(i, 0)].clone();
        }

        return (ipiv, LuInfo::Ok);
    }

    // we're splitting a up into the block matrix
    // [ a11 a12 ]
    // [ a21 a22 ]

    let min_dimension = usize::min(m, n);
    let n1 = min_dimension / 2;
    let n2 = n - n1;

    // recursively solve for
    // [ a11 ]
    // [ a21 ]
    let mut a1 = a.view_mut((.., ..n1));
    let (mut ipiv, info) = lu_driver(&mut a1);

    // apply the pivots from that lu to
    // [ a12 ]
    // [ a22 ]
    let mut a2 = a.view_mut((.., n1..));
    for (i, ix) in ipiv.iter().enumerate() {
        a2.swap_rows(i, *ix);
    }

    // find X where a11 * X = a12
    // and replace a12 with X
    for i in 1..n1 {
        for k in 0..i {
            let aik = a[(i, k)].clone();
            if aik == T::zero() {
                continue;
            }

            for j in 0..n2 {
                a[(i, n1 + j)] = a[(i, n1 + j)].clone() - (aik.clone() * a[(k, n1 + j)].clone());
            }
        }
    }

    // set a22 = a22 - a21 * a12
    for i in 0..(m - n1) {
        for l in 0..n1 {
            let ail = a[(n1 + i, l)].clone();
            for j in 0..n2 {
                a[(n1 + i, n1 + j)] =
                    a[(n1 + i, n1 + j)].clone() - (ail.clone() * a[(l, n1 + j)].clone());
            }
        }
    }

    let mut a22 = a.view_mut((n1.., n1..));
    let (ipiv2, info2) = lu_driver(&mut a22);
    ipiv.extend(ipiv2);

    let info = match (info, info2) {
        (LuInfo::ZeroValue(i), _) => LuInfo::ZeroValue(i),
        (LuInfo::Ok, LuInfo::ZeroValue(i)) => LuInfo::ZeroValue(i + n1),
        _ => LuInfo::Ok,
    };

    for pivot in ipiv.iter_mut().take(min_dimension).skip(n1) {
        *pivot += n1;
    }

    let mut a1 = a.view_mut((.., ..n1));
    for (i, idx) in ipiv.iter().enumerate().take(min_dimension).skip(n1) {
        a1.swap_rows(i, *idx);
    }

    (ipiv, info)
}

fn to_pivot_matrix<T: Scalar>(ipiv: &[usize], rows: usize) -> Matrix<T> {
    let mut p = Matrix::identity(rows);
    let mut view = p.as_view_mut();
    for (k, &ix) in ipiv.iter().enumerate() {
        view.swap_rows(k, ix);
    }
    p
}

pub fn lu_factor_mut<T: Scalar + PivotSearch + Clone>(a: &mut Matrix<T>) -> (Vec<usize>, LuInfo) {
    let mut a_view = a.as_view_mut();
    lu_driver(&mut a_view)
}

pub fn lu_factor<T: Scalar + PivotSearch + Clone>(
    a: &Matrix<T>,
) -> (Matrix<T>, Vec<usize>, LuInfo) {
    let mut a = a.clone();
    let (ipiv, info) = {
        let mut a_view = a.as_view_mut();
        lu_driver(&mut a_view)
    };

    (a, ipiv, info)
}

pub fn lu<T: Scalar + PivotSearch + Clone>(
    a: &Matrix<T>,
) -> (Matrix<T>, Matrix<T>, Matrix<T>, LuInfo) {
    let mut a = a.clone();
    let (m, n) = a.shape();
    let k = usize::min(m, n);

    let (ipiv, info) = {
        let mut a_view = a.as_view_mut();
        lu_driver(&mut a_view)
    };
    let p = to_pivot_matrix::<T>(&ipiv, m);

    let u = a.triu(0).submatrix((..k, ..));
    let mut l = a.tril(-1).submatrix((.., ..k));

    for i in 0..k {
        l[(i, i)] = T::one();
    }

    (p, l, u, info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::try_mul;
    use crate::numerical::Tolerance;

    fn lu_tolerance(a: &Matrix<f64>) -> Tolerance<f64> {
        let (m, n) = a.shape();
        let scale = 16.0 * f64::EPSILON * (m.max(n) as f64);
        Tolerance::default()
            .with_relative(scale)
            .with_absolute(scale * f64::max(a.max_norm(), 1.0))
    }

    #[test]
    fn test_lu_identity() {
        let a = Matrix::<f64>::identity(5);
        let (p, l, u, info) = lu(&a);

        assert_eq!(info, LuInfo::Ok);
        assert_eq!(p, a);
        assert_eq!(l, a);
        assert_eq!(u, a);
    }

    #[test]
    fn test_lu_zeros() {
        let a = Matrix::<f64>::zeros((5, 5));
        let (p, l, u, info) = lu(&a);

        assert_eq!(info, LuInfo::ZeroValue(0));
        assert_eq!(p, Matrix::<f64>::identity(5));
        assert_eq!(l, Matrix::<f64>::identity(5));
        assert_eq!(u, a);
    }

    #[test]
    fn test_lu_singular() {
        let a = Matrix::<f64>::try_from_iter([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 0.0],
        ])
        .unwrap();
        let (p, l, u, info) = lu(&a);

        assert_eq!(info, LuInfo::ZeroValue(2));

        let tolerance = lu_tolerance(&a);
        let pa = try_mul(p, a).unwrap();
        let lu = try_mul(l, u).unwrap();
        assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
    }

    #[test]
    fn test_lu_tall() {
        let a =
            Matrix::<f64>::try_from_iter([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]]).unwrap();
        let (p, l, u, info) = lu(&a);
        assert_eq!(info, LuInfo::Ok);

        assert_eq!(p.shape(), (4, 4));
        assert_eq!(l.shape(), (4, 2));
        assert_eq!(u.shape(), (2, 2));

        let tolerance = lu_tolerance(&a);
        let pa = try_mul(p, a).unwrap();
        let lu = try_mul(l, u).unwrap();
        assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
    }

    #[test]
    fn test_lu_wide() {
        let a = Matrix::<f64>::try_from_iter([[1.0, 2.0, 3.0, 4.0], [6.0, 7.0, 8.0, 9.0]]).unwrap();
        let (p, l, u, info) = lu(&a);
        assert_eq!(info, LuInfo::Ok);

        assert_eq!(p.shape(), (2, 2));
        assert_eq!(l.shape(), (2, 2));
        assert_eq!(u.shape(), (2, 4));

        let tolerance = lu_tolerance(&a);
        let pa = try_mul(p, a).unwrap();
        let lu = try_mul(l, u).unwrap();
        assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
    }

    #[test]
    fn test_lu_row() {
        let a = Matrix::try_from_iter([[3.0, -1.0, 2.0]]).unwrap();
        let (p, l, u, info) = lu(&a);

        assert_eq!(info, LuInfo::Ok);

        assert_eq!(p, Matrix::<f64>::identity(1));
        assert_eq!(l, Matrix::<f64>::identity(1));
        assert_eq!(u, a);
    }

    #[test]
    fn test_lu_column() {
        let a = Matrix::try_from_iter([[1.0], [5.0], [-3.0]]).unwrap();
        let (p, l, u, info) = lu(&a);

        assert_eq!(info, LuInfo::Ok);

        let tolerance = lu_tolerance(&a);
        let pa = try_mul(p, a).unwrap();
        let lu = try_mul(l, u).unwrap();
        assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_lu_random_square() {
        use rand::{SeedableRng, rngs::SmallRng};
        for n in 1..=24usize {
            for seed in 0..4u64 {
                let mut rng = SmallRng::seed_from_u64(seed);
                let a = Matrix::<f64>::random(&mut rng, n, n);

                let (p, l, u, _) = lu(&a);

                let tolerance = lu_tolerance(&a);
                let pa = try_mul(p, a).unwrap();
                let lu = try_mul(l, u).unwrap();
                assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
            }
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_lu_random_rectangular() {
        use rand::{SeedableRng, rngs::SmallRng};
        for n in 1..=24usize {
            for seed in 0..4u64 {
                let mut rng = SmallRng::seed_from_u64(seed);

                // tall matrices
                {
                    let a = Matrix::<f64>::random(&mut rng, 2 * n + 1, n);
                    let (p, l, u, _) = lu(&a);

                    let tolerance = lu_tolerance(&a);
                    let pa = try_mul(p, a).unwrap();
                    let lu = try_mul(l, u).unwrap();
                    assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
                }

                // wide matrices
                {
                    let a = Matrix::<f64>::random(&mut rng, n, 2 * n + 1);
                    let (p, l, u, _) = lu(&a);

                    let tolerance = lu_tolerance(&a);
                    let pa = try_mul(p, a).unwrap();
                    let lu = try_mul(l, u).unwrap();
                    assert!(Matrix::<f64>::is_close(pa, lu, tolerance));
                }
            }
        }
    }
}
