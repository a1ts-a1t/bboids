mod lu;

use crate::{
    Matrix, Vector,
    arith::Scalar,
    err::LinAlgError,
    linalg::lu::{PivotSearch, lu_driver},
};

pub use crate::linalg::lu::lu;

pub fn solve<T: Scalar + PivotSearch + Clone>(
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

    lu_driver(&mut lu_view);

    todo!()
}
