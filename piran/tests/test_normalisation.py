import math
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran import cpdr
from piran.resonance import replace_cpdr_symbols
from piran.normalisation import compute_root_pairs, compute_glauert_normalisation_factor


class TestNormalisationFactors:
    def setup_method(self):
        q_e = -const.e.si  # Signed electron charge
        q_p = const.e.si  # Signed proton charge

        # Magnetic field
        M = 8.033454e15 * (u.tesla * u.m**3)
        mlat = Angle(0, u.deg)
        l_shell = 4.5 * u.dimensionless_unscaled
        B = (M * math.sqrt(1 + 3 * math.sin(mlat.rad) ** 2)) / (
            l_shell**3 * const.R_earth**3 * math.cos(mlat.rad) ** 6
        )

        frequency_ratio = 1.5 * u.dimensionless_unscaled

        self.Omega_e = (q_e * B) / const.m_e
        Omega_e_abs = abs(self.Omega_e)
        self.omega_pe = Omega_e_abs * frequency_ratio

        n_ = self.omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2
        self.Omega_p = (q_p * B) / const.m_p
        self.omega_pp = np.sqrt((n_ * q_p**2) / (const.eps0 * const.m_p))

        omega_ratio = 0.1225
        self.omega = abs(self.Omega_e) * omega_ratio

    def test_glauert_normalisation_1(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range_glauert_integral = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        glauert_root_pairs = compute_root_pairs(
            self.omega,
            X_range_glauert_integral,
            self.Omega_e,
            self.Omega_p,
            self.omega_pe,
            self.omega_pp,
        )

        dispersion = cpdr.Cpdr(2)
        dispersion.as_poly_in_k()
        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(dispersion._poly_k, values_dict)

        glauert_norm_factor = compute_glauert_normalisation_factor(
            dispersion_poly_k,
            glauert_root_pairs,
            "simpson",
        )

        expected = -8.53830973e-19
        assert math.isclose(glauert_norm_factor, expected, rel_tol=1e-09, abs_tol=1e-27)

    def test_glauert_normalisation_2(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range_glauert_integral = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        glauert_root_pairs = compute_root_pairs(
            self.omega,
            X_range_glauert_integral,
            self.Omega_e,
            self.Omega_p,
            self.omega_pe,
            self.omega_pp,
        )

        dispersion = cpdr.Cpdr(2)
        dispersion.as_poly_in_k()
        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(dispersion._poly_k, values_dict)

        glauert_norm_factor = compute_glauert_normalisation_factor(
            dispersion_poly_k,
            glauert_root_pairs,
            "trapezoid",
        )

        expected = -8.53769363e-19
        assert math.isclose(glauert_norm_factor, expected, rel_tol=1e-09, abs_tol=1e-27)

    def test_glauert_normalisation_3(self):
        X_min = 0.00
        X_max = 5.67
        X_npoints = 300
        X_range_glauert_integral = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        glauert_root_pairs = compute_root_pairs(
            self.omega,
            X_range_glauert_integral,
            self.Omega_e,
            self.Omega_p,
            self.omega_pe,
            self.omega_pp,
        )

        dispersion = cpdr.Cpdr(2)
        dispersion.as_poly_in_k()
        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(dispersion._poly_k, values_dict)

        glauert_norm_factor = compute_glauert_normalisation_factor(
            dispersion_poly_k,
            glauert_root_pairs,
            "simpson",
        )

        expected = -8.87151913e-19
        assert math.isclose(glauert_norm_factor, expected, rel_tol=1e-09, abs_tol=1e-27)
