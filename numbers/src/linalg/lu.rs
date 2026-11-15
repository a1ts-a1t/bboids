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

// this is taken directly from dgetrf2
pub enum LuInfo {
    Ok,
    ZeroValue(usize),
}

/// returns the pivots if successful
/// copying off of dgetrf2
/// https://netlib.org/lapack/explore-html/d8/d2f/group__getrf2_gaebbbe6a533dfea4454123e0553bd3575.html
pub(crate) fn lu_factor_mut<T: Scalar + PivotSearch + Clone>(
    a: &mut ViewMut<T>,
) -> (Vec<usize>, LuInfo) {
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
    let (mut ipiv, info) = lu_factor_mut(&mut a1);

    // apply the pivots from that lu to
    // [ a12 ]
    // [ a22 ]
    let mut a2 = a.view_mut((.., n1..));
    for (i, ix) in ipiv.iter().enumerate() {
        a2.swap_rows(i, *ix); // i think this is what dlaswp does?
    }

    // find X where a11 * X = a12
    // and replace a12 with X
    for j in 0..n2 {
        for k in 0..n1 {
            let bkj = a[(k, n1 + j)].clone();
            if bkj == T::zero() {
                continue;
            }

            for i in k + 1..n1 {
                a[(i, n1 + j)] = a[(i, n1 + j)].clone() + (bkj.clone() * a[(i, k)].clone()).neg();
            }
        }
    }

    // set a22 = a22 - a21 * a12
    // m, n = n2; k = n1
    for j in 0..n2 {
        for l in 0..n1 {
            let blj = a[(l, n1 + j)].clone();
            for i in 0..n2 {
                a[(n1 + i, n1 + j)] =
                    a[(n1 + i, n1 + j)].clone() + (blj.clone() * a[(n1 + i, l)].clone()).neg();
            }
        }
    }

    let mut a22 = a.view_mut((n1.., n1..));
    let (ipiv2, info2) = lu_factor_mut(&mut a22);
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
