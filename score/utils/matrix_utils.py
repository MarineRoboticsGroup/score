import numpy as np
import scipy.linalg as la  # type: ignore
from typing import List, Tuple, Optional


def apply_transformation_matrix_perturbation(
    transformation_matrix,
    perturb_magnitude: Optional[float],
    perturb_rotation: Optional[float],
) -> np.ndarray:
    """Applies a random SE(2) perturbation to a transformation matrix

    Args:
        transformation_matrix ([type]): [description]
        perturb_magnitude (Optional[float]): [description]
        perturb_rotation (Optional[float]): [description]

    Returns:
        np.ndarray: [description]
    """
    _check_transformation_matrix(transformation_matrix)

    # get the x/y perturbation
    perturb_direction = np.random.uniform(0, 2 * np.pi)
    perturb_x = np.cos(perturb_direction) * perturb_magnitude
    perturb_y = np.sin(perturb_direction) * perturb_magnitude

    # get the rotation perturbation
    perturb_theta = np.random.choice([-1, 1]) * perturb_rotation

    # compose the perturbation into a transformation matrix
    rand_trans = np.eye(3)
    rand_trans[:2, :2] = get_rotation_matrix_from_theta(perturb_theta)
    rand_trans[:2, 2] = perturb_x, perturb_y
    _check_transformation_matrix(rand_trans)

    # perturb curr pose
    return transformation_matrix @ rand_trans


def round_to_special_orthogonal(mat: np.ndarray) -> np.ndarray:
    """
    Rounds a matrix to special orthogonal form.

    Args:
        mat (np.ndarray): the matrix to round

    Returns:
        np.ndarray: the rounded matrix
    """
    _check_square(mat)
    _check_rotation_matrix(mat, assert_test=False)
    S, D, Vh = la.svd(mat)
    R_so = S @ Vh
    if np.linalg.det(R_so) < 0:
        R_so = S @ np.diag([1, -1]) @ Vh
    _check_rotation_matrix(R_so, assert_test=True)
    return R_so


def get_theta_from_rotation_matrix_so_projection(mat: np.ndarray) -> float:
    """
    Returns theta from the projection of the matrix M onto the special
    orthogonal group

    Args:
        mat (np.ndarray): the candidate rotation matrix

    Returns:
        float: theta

    """
    R_so = round_to_special_orthogonal(mat)
    return get_theta_from_rotation_matrix(R_so)


def get_theta_from_rotation_matrix(mat: np.ndarray) -> float:
    """
    Returns theta from a matrix M

    Args:
        mat (np.ndarray): the candidate rotation matrix

    Returns:
        float: theta
    """
    _check_square(mat)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def get_random_vector(dim: int) -> np.ndarray:
    """Returns a random vector of size dim

    Args:
        dim (int): the dimension of the vector

    Returns:
        np.ndarray: the random vector
    """
    return np.random.rand(dim)


def get_rotation_matrix_from_theta(theta: float) -> np.ndarray:
    """Returns the rotation matrix from theta

    Args:
        theta (float): the angle of rotation

    Returns:
        np.ndarray: the rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def get_rotation_matrix_from_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """Returns the rotation matrix from the transformation matrix

    Args:
        T (np.ndarray): the transformation matrix

    Returns:
        np.ndarray: the rotation matrix
    """
    return T[:2, :2]


def get_theta_from_transformation_matrix(T: np.ndarray) -> float:
    """Returns the angle theta from a transformation matrix

    Args:
        T (np.ndarray): the transformation matrix

    Returns:
        float: the angle theta
    """
    return get_theta_from_rotation_matrix(
        get_rotation_matrix_from_transformation_matrix(T)
    )


def get_translation_from_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """Returns the translation from a transformation matrix

    Args:
        T (np.ndarray): the transformation matrix

    Returns:
        np.ndarray: the translation
    """
    return T[:2, 2]


def get_random_rotation_matrix(dim: int = 2) -> np.ndarray:
    """Returns a random rotation matrix of size dim x dim"""
    if dim == 2:
        theta = 2 * np.pi * np.random.rand()
        return get_rotation_matrix_from_theta(theta)
    else:
        raise NotImplementedError("Only implemented for dim = 2")


def get_random_transformation_matrix(dim: int = 2) -> np.ndarray:
    if dim == 2:
        R = get_random_rotation_matrix(dim)
        t = get_random_vector(dim)
        return make_transformation_matrix(R, t)
    else:
        raise NotImplementedError("Only implemented for dim = 2")


def make_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Returns the transformation matrix from a rotation matrix and translation vector

    Args:
        R (np.ndarray): the rotation matrix
        t (np.ndarray): the translation vector

    Returns:
        np.ndarray: the transformation matrix
    """
    _check_rotation_matrix(R)
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    _check_transformation_matrix(T)
    return T


def make_transformation_matrix_from_theta(
    theta: float,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Returns the transformation matrix from theta and translation

    Args:
        theta (float): the angle of rotation
        translation (np.ndarray): the translation

    Returns:
        np.ndarray: the transformation matrix
    """
    R = get_rotation_matrix_from_theta(theta)
    return make_transformation_matrix(R, translation)


#### test functions ####


def _check_rotation_matrix(R: np.ndarray, assert_test: bool = False):
    """
    Checks that R is a rotation matrix.

    Args:
        R (np.ndarray): the candidate rotation matrix
        assert_test (bool): if false just print if not rotation matrix, otherwise raise error
    """
    d = R.shape[0]
    is_orthogonal = np.allclose(R @ R.T, np.eye(d), rtol=1e-3, atol=1e-3)
    if not is_orthogonal:
        # print(f"R not orthogonal: {R @ R.T}")
        if assert_test:
            raise ValueError(f"R is not orthogonal {R @ R.T}")

    has_correct_det = abs(np.linalg.det(R) - 1) < 1e-3
    if not has_correct_det:
        # print(f"R det != 1: {np.linalg.det(R)}")
        if assert_test:
            print(la.svd(R))
            print(la.eigvals(R))
            print(R)
            raise ValueError(f"R det incorrect {np.linalg.det(R)}")
    # print(f"R is a rotation matrix {R}")
    # print()


def _check_square(mat: np.ndarray):
    assert mat.shape[0] == mat.shape[1], "matrix must be square"


def _check_symmetric(mat):
    assert np.allclose(mat, mat.T)


def _check_psd(mat: np.ndarray):
    """Checks that a matrix is positive semi-definite"""
    assert isinstance(mat, np.ndarray)
    assert (
        np.min(la.eigvals(mat)) + 1e-1 >= 0.0
    ), f"min eigenvalue is {np.min(la.eigvals(mat))}"


def _check_is_laplacian(L: np.ndarray):
    """Checks that a matrix is a Laplacian based on well-known properties

    Must be:
        - symmetric
        - ones vector in null space of L
        - no negative eigenvalues

    Args:
        L (np.ndarray): the candidate Laplacian
    """
    assert isinstance(L, np.ndarray)
    _check_symmetric(L)
    _check_psd(L)
    ones = np.ones(L.shape[0])
    zeros = np.zeros(L.shape[0])
    assert np.allclose(L @ ones, zeros), f"L @ ones != zeros: {L @ ones}"


def _check_transformation_matrix(T: np.ndarray, assert_test: bool = True):
    """Checks that the matrix passed in is a homogeneous transformation matrix.
    If assert_test is True, then this is in the form of assertions, otherwise we
    just print out error messages but continue

    Args:
        T (np.ndarray): the matrix to test
        assert_test (bool, optional): Whether this is a 'hard' test and is
        asserted or just a 'soft' test and only prints message if test fails. Defaults to True.
    """
    _check_square(T)
    assert (
        T.shape[0] == 3
    ), f"only considering 2d world right now so matrix must be 3x3, received {T.shape}"

    # check that is rotation matrix in upper left block
    R = T[0:2, 0:2]
    _check_rotation_matrix(R, assert_test=assert_test)

    # check that the bottom row is [0, 0, 1]
    bottom = T[2, :]
    bottom_expected: np.ndarray = np.ndarray([0, 0, 1])
    assert np.allclose(bottom.flatten(), bottom_expected)


#### print functions ####


def _print_eigvals(
    M: np.ndarray, name: str = None, print_eigvec: bool = False, symmetric: bool = True
):
    """print the eigenvalues of a matrix"""

    if name is not None:
        print(name)

    if print_eigvec:
        # get the eigenvalues of the matrix
        if symmetric:
            eigvals, eigvecs = la.eigh(M)
        else:
            eigvals, eigvecs = la.eig(M)

        # sort the eigenvalues and eigenvectors
        idx = eigvals.argsort()[::1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        print(f"eigenvectors: {eigvecs}")
    else:
        if symmetric:
            eigvals = la.eigvalsh(M)
        else:
            eigvals = la.eigvals(M)
        print(f"eigenvalues\n{eigvals}")

    print("\n\n\n")


def _matprint_block(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    num_col = mat.shape[1]
    row_spacer = ""
    for _ in range(num_col):
        row_spacer += "__ __ __ "
    for j, x in enumerate(mat):
        if j % 2 == 0:
            print(row_spacer)
            print("")
        for i, y in enumerate(x):
            if i % 2 == 1:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end=" | ")
            else:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")

    print(row_spacer)
    print("\n\n\n")
