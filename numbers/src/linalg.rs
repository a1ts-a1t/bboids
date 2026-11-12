use crate::{Matrix, Vector, arith::Scalar, err::LinAlgError};

pub fn solve<T: Scalar>(a: Matrix<T>, b: Vector<T>) -> Result<Vector<T>, LinAlgError> {
    if !a.is_square() {
        return Err(LinAlgError::IncompatibleShape);
    }

    let (n, _) = a.shape();
    if n == 0 || n != b.shape() {
        return Err(LinAlgError::IncompatibleShape);
    }

    todo!()
}
