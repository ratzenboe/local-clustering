import numpy as np
import sympy as sp


TGAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ]
)

# Initiate some global constants
# 1 AU/yr to km/s divided by 1000
kappa = 0.004743717361

r1 = sp.symbols('r1')
r2 = sp.symbols('r2')


def create_vector(
        ra,
        dec,
        pmra,
        pmdec,
        dist,
        rv = r1):
    """
    Transforms equatorial coordinates (ra,dec), proper motion (pmra,pmdec), radial velocity and distance to space velocities UVW.
    All inputs must be numpy arrays of the same dimension.

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    param pmra: Proper motion in right ascension (milliarcsecond per year). Must include the cos(delta) term!
    param pmdec: Proper motion in declination (milliarcsecond per year)
    param rv: Radial velocity (kilometers per second) make sure to set this to r1 and r2 when comparing two objects!
    param dist: Distance (parsec)

    output (U,V,W): Tuple containing Space velocities UVW (kilometers per second)
    """
    # Compute elements of the T matrix
    cos_ra = np.cos(np.radians(ra))
    cos_dec = np.cos(np.radians(dec))
    sin_ra = np.sin(np.radians(ra))
    sin_dec = np.sin(np.radians(dec))

    T1 = (TGAL[0, 0] * cos_ra * cos_dec + TGAL[0, 1] * sin_ra * cos_dec + TGAL[0, 2] * sin_dec)
    T2 = -TGAL[0, 0] * sin_ra + TGAL[0, 1] * cos_ra
    T3 = (-TGAL[0, 0] * cos_ra * sin_dec - TGAL[0, 1] * sin_ra * sin_dec + TGAL[0, 2] * cos_dec)
    T4 = (TGAL[1, 0] * cos_ra * cos_dec + TGAL[1, 1] * sin_ra * cos_dec + TGAL[1, 2] * sin_dec)
    T5 = -TGAL[1, 0] * sin_ra + TGAL[1, 1] * cos_ra
    T6 = (-TGAL[1, 0] * cos_ra * sin_dec - TGAL[1, 1] * sin_ra * sin_dec + TGAL[1, 2] * cos_dec)
    T7 = (TGAL[2, 0] * cos_ra * cos_dec + TGAL[2, 1] * sin_ra * cos_dec + TGAL[2, 2] * sin_dec)
    T8 = -TGAL[2, 0] * sin_ra + TGAL[2, 1] * cos_ra
    T9 = (-TGAL[2, 0] * cos_ra * sin_dec - TGAL[2, 1] * sin_ra * sin_dec + TGAL[2, 2] * cos_dec)

    reduced_dist = kappa * dist

    TM = np.array(
        [
            [T1, T2, T3],
            [T4, T5, T6],
            [T7, T8, T9]
        ]
    )

    rad_vector = np.array(
        [rv, pmra * reduced_dist, pmdec * reduced_dist]
    )
    xyz_vector = np.dot(TM, rad_vector)
    # U = xyz_vector[0]
    # V = xyz_vector[1]
    # W = xyz_vector[2]
    return xyz_vector


def vector_diff(v1, v2):
    diff = v1-v2
    norm2 = diff[0]**2 + diff[1]**2 + diff[2]**2
    return norm2


def get_gradient(norm):
    partial_r1 = sp.diff(norm, r1)
    partial_r2 = sp.diff(norm, r2)
    grad = np.array(
        [partial_r1, partial_r2]
    )
    coeffs1 = sp.Poly(grad[0], r1, r2).coeffs()
    constant1 = coeffs1.pop()

    coeffs2 = sp.Poly(grad[1], r1, r2).coeffs()
    constant2 = coeffs2.pop()
    A = np.array(
        [
            coeffs1,
            coeffs2
        ], dtype=float
    )
    b = np.array([constant1, constant2], dtype=float)

    solution = np.linalg.solve(A, b)

    return solution
