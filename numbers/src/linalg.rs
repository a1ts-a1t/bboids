use crate::{
    Matrix, Vector,
    arith::{OrderedScalar, Scalar},
    err::LinAlgError,
    matrix::ViewMut,
};

// TODO
// this is taken directly from dgetrf2
// https://netlib.org/lapack/explore-html/d8/d2f/group__getrf2_gaebbbe6a533dfea4454123e0553bd3575.html
// we probably want to make this more... rusty
enum LuInfo {
    Ok,
    ZeroValue(usize),
}

/// returns the pivots if successful
/// copying off of dgetrf2
/// https://netlib.org/lapack/explore-html/d8/d2f/group__getrf2_gaebbbe6a533dfea4454123e0553bd3575.html
fn lu_factor_mut<T: OrderedScalar + Clone>(a: &mut ViewMut<T>) -> (Vec<usize>, LuInfo) {
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
        let (argmax_index, _) = a.column(0).argmax().unwrap();
        let ipiv = [argmax_index].to_vec();

        if a[(argmax_index, 0)] == T::zero() {
            return (ipiv, LuInfo::ZeroValue(0));
        }

        if argmax_index != 0 {
            a.swap((0, 0), (argmax_index, 0));
        }

        // TODO: is there any way to do this without clone?
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
    // resolving into a matrix?
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

    for i in n1..min_dimension {
        ipiv[i] += n1;
    }

    let mut a1 = a.view_mut((.., ..n1));
    for i in n1..min_dimension {
        let idx = ipiv[i];
        a1.swap_rows(i, idx);
    }

    (ipiv, info)
}

pub fn lu<T: Scalar + Clone>(
    a: Matrix<T>,
) -> Result<(Matrix<T>, Matrix<T>, Matrix<T>), LinAlgError> {
    todo!()
}

pub fn solve<T: OrderedScalar + Clone>(
    a: Matrix<T>,
    b: Vector<T>,
) -> Result<Vector<T>, LinAlgError> {
    if !a.is_square() {
        return Err(LinAlgError::IncompatibleShape);
    }

    let (n, _) = a.shape();
    if n == 0 || n != b.shape() {
        return Err(LinAlgError::IncompatibleShape);
    }

    let mut lu = a.clone();
    let mut lu_view = lu.as_view_mut();

    lu_factor_mut(&mut lu_view);

    todo!()
}
