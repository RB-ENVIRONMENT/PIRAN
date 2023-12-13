import math
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.magfield import MagField
from piran.particles import Particles
from piran.particles import PiranParticle
from piran.gauss import Gaussian
from piran.resonance import replace_cpdr_symbols
from piran.normalisation import (
    solve_dispersion_relation,
    compute_glauert_normalisation_factor,
)


class TestNormalisationFactors:
    def setup_method(self):
        frequency_ratio = 1.5 * u.dimensionless_unscaled
        omega_ratio = 0.1225

        # ============================== START ============================== #
        # Those should be attributes of one of the main classes
        # Magnetic field
        M = 8.033454e15 * (u.tesla * u.m**3)
        mlat = Angle(0, u.deg)
        l_shell = 4.5 * u.dimensionless_unscaled
        B = (M * math.sqrt(1 + 3 * math.sin(mlat.rad) ** 2)) / (
            l_shell**3 * const.R_earth**3 * math.cos(mlat.rad) ** 6
        )

        q_e = -const.e.si  # Signed electron charge
        q_p = const.e.si  # Signed proton charge

        self.Omega_e = (q_e * B) / const.m_e
        Omega_e_abs = abs(self.Omega_e)
        self.omega_pe = Omega_e_abs * frequency_ratio

        n_ = self.omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2
        self.Omega_p = (q_p * B) / const.m_p
        self.omega_pp = np.sqrt((n_ * q_p**2) / (const.eps0 * const.m_p))
        # =============================== END =============================== #

        # ============================== START ============================== #
        # We need those because they are input arguments to the new Cpdr class.
        # They are not needed for these tests.
        RKE = 1.0 * u.MeV  # Relativistic kinetic energy (Mega-electronvolts)
        alpha = Angle(5, u.deg)  # pitch angle

        # Lower and upper cut-off frequencies
        omega_m = 0.35 * Omega_e_abs
        delta_omega = 0.15 * Omega_e_abs
        omega_lc = omega_m - 1.5 * delta_omega
        omega_uc = omega_m + 1.5 * delta_omega

        # Resonances
        n_min = -5
        n_max = 5
        n_range = u.Quantity(
            range(n_min, n_max + 1), unit=u.dimensionless_unscaled, dtype=np.int32
        )
        # =============================== END =============================== #

        self.omega = abs(self.Omega_e) * omega_ratio

        piran_particle_list = (PiranParticle("e", n_), PiranParticle("H+", n_))
        cpdr_particles = Particles(piran_particle_list, RKE, alpha)
        # NOTE upper is just a very large number for now (X_max?)
        cpdr_wave_angles = Gaussian(0, 1e10, 0, 0.577)
        cpdr_wave_freqs = Gaussian(omega_lc, omega_uc, omega_m, delta_omega)
        cpdr_mag_field = MagField(mlat, l_shell)
        cpdr_resonances = n_range

        self.dispersion = Cpdr(
            cpdr_particles,
            cpdr_wave_angles,
            cpdr_wave_freqs,
            cpdr_mag_field,
            cpdr_resonances,
        )

        self.dispersion.as_poly_in_k()

    def test_glauert_normalisation_1(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range_glauert_integral = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        glauert_root_pairs = solve_dispersion_relation(
            self.dispersion,
            (self.Omega_e, self.Omega_p),
            (self.omega_pe, self.omega_pp),
            self.omega,
            X_range_glauert_integral,
        )

        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(self.dispersion._poly_k, values_dict)

        glauert_norm_factor = compute_glauert_normalisation_factor(
            dispersion_poly_k,
            glauert_root_pairs,
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

        glauert_root_pairs = solve_dispersion_relation(
            self.dispersion,
            (self.Omega_e, self.Omega_p),
            (self.omega_pe, self.omega_pp),
            self.omega,
            X_range_glauert_integral,
        )

        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(self.dispersion._poly_k, values_dict)

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

        glauert_root_pairs = solve_dispersion_relation(
            self.dispersion,
            (self.Omega_e, self.Omega_p),
            (self.omega_pe, self.omega_pp),
            self.omega,
            X_range_glauert_integral,
        )

        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(self.dispersion._poly_k, values_dict)

        glauert_norm_factor = compute_glauert_normalisation_factor(
            dispersion_poly_k,
            glauert_root_pairs,
        )

        expected = -8.87151913e-19
        assert math.isclose(glauert_norm_factor, expected, rel_tol=1e-09, abs_tol=1e-27)
