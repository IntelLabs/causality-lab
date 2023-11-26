import numpy as np


def cov_to_corr(cov_matrix: np.ndarray):
    """
    Convert a square covariance matrix (COV) into a correlation matrix. An element (i, j) of the correlation matrix is
    equal to COV(i,j) / sqrt(COV(i,i)COV(j,j)).

    :param cov_matrix: input covariance matrix in numpy.ndarray format
    :return: correlation matrix.
    """
    assert cov_matrix.ndim == 2  # matrix
    assert cov_matrix.shape[0] == cov_matrix.shape[1]  # square
    min_variance = 1e-8  # minimal variance allowed
    diag = np.sqrt(np.diag(cov_matrix))
    assert np.all(diag > min_variance)  # variance is smaller than the supported minimal value
    inv_std = 1.0 / diag

    # Calculate the correlation matrix
    correlation_matrix = cov_matrix * inv_std * inv_std[:, np.newaxis]  # Cor = (D^-1) @ Cov @ (D^-1), D=sqrt(diag(Cov))

    return correlation_matrix
