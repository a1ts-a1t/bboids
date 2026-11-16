use crate::{
    Matrix, Vector,
    arith::Scalar,
    err::LinAlgError,
    linalg::lu::{LuInfo, PivotSearch, lu_factor},
};

pub fn lu_solve_mut<T: Scalar + PivotSearch + Clone>(
    lu: &Matrix<T>,
    ipiv: &[usize],
    b: &mut Vector<T>,
) -> Result<(), LinAlgError> {
    if !lu.is_square() {
        return Err(LinAlgError::IncompatibleShape);
    }

    let (n, _) = lu.shape();
    if n == 0 || n != b.shape() {
        return Err(LinAlgError::IncompatibleShape);
    };

    // dgetrs
    // https://netlib.org/lapack/explore-html/df/d36/group__getrs_gaacd7a8465c8cc0e4e8b88ba4b453630c.html
    for (i, ix) in ipiv.iter().enumerate() {
        b.swap(i, *ix);
    }

    // find x where L * x = b and replace b with x
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..i {
            sum = sum + (b[j].clone() * lu[(i, j)].clone());
        }
        b[i] = b[i].clone() - sum;
    }

    // do the same now with upper
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for j in i + 1..n {
            sum = sum + (b[j].clone() * lu[(i, j)].clone());
        }
        b[i] = (b[i].clone() - sum) * lu[(i, i)].inverse().ok_or(LinAlgError::Singular)?;
    }

    Ok(())
}

pub fn lu_solve<T: Scalar + PivotSearch + Clone>(
    lu: &Matrix<T>,
    ipiv: &[usize],
    b: &Vector<T>,
) -> Result<Vector<T>, LinAlgError> {
    let mut b = b.clone();
    lu_solve_mut(lu, ipiv, &mut b)?;
    Ok(b)
}

pub fn solve<T: Scalar + PivotSearch + Clone>(
    a: &Matrix<T>,
    b: &Vector<T>,
) -> Result<Vector<T>, LinAlgError> {
    // some early shape checks. quick enough to repeat
    if !a.is_square() {
        return Err(LinAlgError::IncompatibleShape);
    }

    let (n, _) = a.shape();

    if n == 0 || n != b.shape() {
        return Err(LinAlgError::IncompatibleShape);
    };

    let (lu, ipiv, info) = lu_factor(a);

    if matches!(info, LuInfo::ZeroValue(_)) {
        return Err(LinAlgError::Singular);
    }

    lu_solve(&lu, &ipiv, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    #[test]
    fn test_solve_identity() {
        let i = Matrix::<f64>::identity(5);
        let b = Vector::<f64>::from_iter([1.0, 2.0, 3.0, 4.0, 5.0]);

        let actual = solve(&i, &b).unwrap();
        assert_eq!(b, actual);
    }

    #[test]
    fn test_incompatible_shape() {
        let a = Matrix::<f64>::try_from_iter([[1.0, 1.0], [2.0, 2.0]]).unwrap();
        let b = Vector::<f64>::from_iter([1.0, 2.0, 3.0]);

        let result = solve(&a, &b);

        assert_matches!(result.unwrap_err(), LinAlgError::IncompatibleShape);
    }

    #[test]
    fn test_solve_singular() {
        let a = Matrix::<f64>::try_from_iter([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 0.0],
        ])
        .unwrap();
        let b = Vector::<f64>::from_iter([1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = solve(&a, &b);

        assert_matches!(result.unwrap_err(), LinAlgError::Singular);
    }
}
