"""Affine matrix decomposition and composition utilities."""

import math

import numpy as np
import numpy.typing as npt


def _striu2mat(
    striu: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Construct a shear matrix from the upper triangular vector.

    Parameters
    ----------
    striu : (N,) numpy.ndarray
        Vector giving triangle above diagonal of shear matrix. ``N`` must be a
        triangular number (1, 3, 6, 10, …).

    Returns
    -------
    SM : (M, M) numpy.ndarray
        Shear matrix, where ``M`` is the integer satisfying ``M*(M-1)/2 == N``.

    Raises
    ------
    ValueError
        If ``len(striu)`` is not a triangular number.

    Notes
    -----
    Adapted from ``transforms3d.affines.striu2mat`` by Matthew Brett et al.
    (BSD-2-Clause License). See the ``NOTICE`` and ``LICENSE-BSD-2-Clause``
    files for details.
    Source: https://github.com/matthew-brett/transforms3d
    """
    n = len(striu)
    N = (-1 + math.sqrt(8 * n + 1)) / 2.0 + 1
    if N != math.floor(N):
        raise ValueError(f"{n} is a strange number of shear elements")
    N = int(N)
    M = np.eye(N)
    inds = np.triu(np.ones((N, N)), 1).astype(bool)
    M[inds] = striu
    return M


def compose_affine(
    T: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    Z: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compose translations, rotations, zooms, and shears into an affine matrix.

    Parameters
    ----------
    T : (N,) numpy.ndarray
        Translation vector, where ``N`` is usually 3 (3D case).
    R : (N, N) numpy.ndarray
        Rotation matrix, where ``N`` is usually 3 (3D case).
    Z : (N,) numpy.ndarray
        Zoom (scale) vector, where ``N`` is usually 3 (3D case).
    S : (P,) numpy.ndarray, optional
        Shear vector filling the upper triangle above the diagonal of the shear
        matrix. ``P`` is the ``(N-2)``-th triangular number (3 for the 3D case).
        If ``None``, no shear is applied.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Homogeneous affine transformation matrix.

    Notes
    -----
    Adapted from ``transforms3d.affines.compose`` by Matthew Brett _et al_.
    (BSD-2-Clause License). See the ``NOTICE`` and ``LICENSE-BSD-2-Clause``
    files for details.
    Source: https://github.com/matthew-brett/transforms3d
    """
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError(f"Expected shape ({n},{n}) for R, got {R.shape}")
    A = np.eye(n + 1)
    ZS = np.diag(Z)
    if S is not None:
        ZS = ZS @ _striu2mat(S)
    A[:n, :n] = R @ ZS
    A[:n, n] = T
    return A


def decompose_affine(
    A44: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Decompose a 4x4 homogeneous affine into translation, rotation, zoom, shear.

    Decomposes ``A44`` into ``T, R, Z, S`` such that::

        Smat = np.array([[1, S[0], S[1]],
                         [0,    1, S[2]],
                         [0,    0,    1]])
        RZS = R @ np.diag(Z) @ Smat
        A44[:3, :3] = RZS
        A44[:3,  3] = T

    Parameters
    ----------
    A44 : (4, 4) numpy.ndarray
        Homogeneous affine matrix.

    Returns
    -------
    T : (3,) numpy.ndarray
        Translation vector.
    R : (3, 3) numpy.ndarray
        Rotation matrix.
    Z : (3,) numpy.ndarray
        Zoom (scale) vector. May have one negative zoom to avoid a negative
        determinant in ``R``.
    S : (3,) numpy.ndarray
        Shear vector ``[sxy, sxz, syz]`` filling the upper triangle of the
        shear matrix. Zero for pure rotation/zoom affines.

    Notes
    -----
    Adapted from ``transforms3d.affines.decompose44`` by Matthew Brett et al.
    (BSD-2-Clause License). See the ``NOTICE`` and ``LICENSE-BSD-2-Clause``
    files for details.
    Source: https://github.com/matthew-brett/transforms3d

    See also: Spencer W. Thomas, "Decomposing a matrix into simple
    transformations", pp 320-323 in *Graphics Gems II*, James Arvo (ed.),
    Academic Press, 1991.
    """
    A44 = np.asarray(A44)
    T = A44[:-1, -1]
    RZS = A44[:-1, :-1]
    M0, M1, M2 = np.array(RZS).T
    sx = math.sqrt(np.sum(M0**2))
    M0 /= sx
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    sy = math.sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= sx_sxz * M0 + sy_syz * M1
    sz = math.sqrt(np.sum(M2**2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    Rmat = np.array([M0, M1, M2]).T
    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:, 0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])
