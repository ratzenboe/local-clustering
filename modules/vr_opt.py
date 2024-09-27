import numpy as np

# Galactic Coordinates matrix
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


class VrOpt:
    def __init__(self, ra, dec, pmra, pmdec, dist, rv=None):
        self.ra = ra
        self.dec = dec
        self.pmra = pmra
        self.pmdec = pmdec
        self.dist = dist
        self.reduced_dist = kappa * self.dist
        self.rv = rv
        self.T_mtx = self.create_trafo_matrix()
        self.const_vec = self.uvw_const_vector()

    def create_trafo_matrix(self):
        """Compute elements of the T matrix"""
        # compute cos and sin of ra and dec
        cos_ra = np.cos(np.radians(self.ra))
        cos_dec = np.cos(np.radians(self.dec))
        sin_ra = np.sin(np.radians(self.ra))
        sin_dec = np.sin(np.radians(self.dec))
        # Compute elements of the T matrix
        T1 = TGAL[0, 0] * cos_ra * cos_dec + TGAL[0, 1] * sin_ra * cos_dec + TGAL[0, 2] * sin_dec
        T2 = -TGAL[0, 0] * sin_ra + TGAL[0, 1] * cos_ra
        T3 = -TGAL[0, 0] * cos_ra * sin_dec - TGAL[0, 1] * sin_ra * sin_dec + TGAL[0, 2] * cos_dec
        T4 = TGAL[1, 0] * cos_ra * cos_dec + TGAL[1, 1] * sin_ra * cos_dec + TGAL[1, 2] * sin_dec
        T5 = -TGAL[1, 0] * sin_ra + TGAL[1, 1] * cos_ra
        T6 = -TGAL[1, 0] * cos_ra * sin_dec - TGAL[1, 1] * sin_ra * sin_dec + TGAL[1, 2] * cos_dec
        T7 = TGAL[2, 0] * cos_ra * cos_dec + TGAL[2, 1] * sin_ra * cos_dec + TGAL[2, 2] * sin_dec
        T8 = -TGAL[2, 0] * sin_ra + TGAL[2, 1] * cos_ra
        T9 = -TGAL[2, 0] * cos_ra * sin_dec - TGAL[2, 1] * sin_ra * sin_dec + TGAL[2, 2] * cos_dec
        # Create the T matrix
        T_total = np.array(
            [
                [T1, T2, T3],
                [T4, T5, T6],
                [T7, T8, T9]
            ]
        )
        return T_total.transpose(2, 0, 1)

    def uvw_const_vector(self):
        """Compute the constant part of the UVW vector"""
        T2 = self.T_mtx[:, 0, 1]
        T3 = self.T_mtx[:, 0, 2]
        T5 = self.T_mtx[:, 1, 1]
        T6 = self.T_mtx[:, 1, 2]
        T8 = self.T_mtx[:, 2, 1]
        T9 = self.T_mtx[:, 2, 2]
        # Compute known part of the UVW vector
        c_u = T2 * self.pmra * self.reduced_dist + T3 * self.pmdec * self.reduced_dist
        c_v = T5 * self.pmra * self.reduced_dist + T6 * self.pmdec * self.reduced_dist
        c_w = T8 * self.pmra * self.reduced_dist + T9 * self.pmdec * self.reduced_dist
        # Return the constant part of the UVW vector
        c_vec = np.array([c_u, c_v, c_w])
        return c_vec.T

    def vr_opt(self, m, n):
        """Compute the optimal radial velocities of two stars which minimizes their 3D velocity difference"""
        # Differences of the constant vectors
        c, f, i = (self.const_vec[m, :] - self.const_vec[n, :]).T
        # Denote elements of the (U_m - U_n)**2 + (V_m - V_n)**2 + (W_m - W_n)**2 vector
        # norm**2 = (a*vr_m - b*vr_n + c)**2 + (d*vr_m - e*vr_n + f)**2 + (g*vr_m - h*vr_n + i)**2
        a = self.T_mtx[:, 0, 0][m]
        b = self.T_mtx[:, 0, 0][n]
        d = self.T_mtx[:, 1, 0][m]
        e = self.T_mtx[:, 1, 0][n]
        g = self.T_mtx[:, 2, 0][m]
        h = self.T_mtx[:, 2, 0][n]
        # Compute optimal radial velocity of star m
        x_num = (
            a*b*e*f + a*b*h*i - a*c*e**2 - a*c*h**2 - b**2 * d*f -
            b**2 *g*i + b*c*d*e + b*c*g*h + d*e*h*i - d*f*h**2 - e**2 *g*i + e*f*g*h
        )
        x_den = (
            a**2 * e**2 + a**2 * h**2 - 2*a*b*d*e - 2*a*b*g*h +
            b**2 * d**2 + b**2 * g**2 + d**2 * h**2 - 2*d*e*g*h + e**2 * g**2
        )
        vr_m = x_num / x_den
        # Compute optimal radial velocity of star n
        y_num = (
            a**2 * e*f + a**2 * h*i - a*b*d*f - a*b*g*i - a*c*d*e - a*c*g*h + b*c*d**2 +
            b*c*g**2 + d**2 * h*i - d*e*g*i - d*f*g*h + e*f*g**2
        )
        y_den = (
            a**2 * e**2 + a**2 * h**2 - 2*a*b*d*e - 2*a*b*g*h +
            b**2 * d**2 + b**2 * g**2 + d**2 * h**2 - 2*d*e*g*h + e**2 * g**2
        )
        vr_n = y_num / y_den
        # Compute the minimal difference of the pairwise 3D velocity vectors
        norm2 = (a*vr_m - b*vr_n + c)**2 + (d*vr_m - e*vr_n + f)**2 + (g*vr_m - h*vr_n + i)**2

        # Compute hessian matrix
        # hessian = np.array(
        #     [
        #         [2*(a**2 + d**2 + g**2), -2*(a*b + d*e + g*h)],
        #         [-2*(a*b + d*e + g*h),  2*(b**2 + e**2 + h**2)],
        #
        #     ]
        # )
        # Return the optimal radial velocities and the minimal difference of the 3D velocity vectors
        return vr_m, vr_n, np.sqrt(norm2)  #, hessian.transpose(2, 0, 1)
