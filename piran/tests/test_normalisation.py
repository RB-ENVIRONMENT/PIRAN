import math

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magfield import MagField
from piran.normalisation import (
    compute_cunningham_normalisation_factor,
    compute_glauert_normalisation_factor,
    solve_dispersion_relation,
)
from piran.particles import Particles, PiranParticle
from piran.resonance import replace_cpdr_symbols


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
        cpdr_mag_field = MagField()
        cpdr_resonances = n_range

        self.dispersion = Cpdr(
            cpdr_particles,
            cpdr_wave_angles,
            cpdr_wave_freqs,
            cpdr_mag_field,
            mlat,
            l_shell,
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
            self.dispersion,
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
            self.dispersion,
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
            self.dispersion,
            dispersion_poly_k,
            glauert_root_pairs,
        )

        expected = -8.87151913e-19
        assert math.isclose(glauert_norm_factor, expected, rel_tol=1e-09, abs_tol=1e-27)

    def test_cunningham_normalisation_1(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        root_pairs = solve_dispersion_relation(
            self.dispersion,
            (self.Omega_e, self.Omega_p),
            (self.omega_pe, self.omega_pp),
            self.omega,
            X_range,
        )

        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(self.dispersion._poly_k, values_dict)

        cnf = compute_cunningham_normalisation_factor(
            dispersion_poly_k,
            root_pairs,
        )

        # Check the first, last, a few points in-between and the sum of all values.
        # If an assertion fails, the test fails and the remaining asserts
        # are not evaluated.
        assert cnf[0] == 0.0
        assert math.isclose(cnf[1], -6.14971590e-20, rel_tol=1e-09, abs_tol=1e-28)
        assert math.isclose(cnf[20], -1.20262320e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf[50], -2.71450691e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf[80], -3.77673193e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf[-1], -4.27878582e-18, rel_tol=1e-09, abs_tol=1e-26)
        assert math.isclose(cnf.sum(), -2.49251344e-16, rel_tol=1e-09, abs_tol=1e-24)

        # Check the size of the array
        assert cnf.size == X_npoints

    def test_jacobian(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        xwk_roots = solve_dispersion_relation(
            self.dispersion,
            (self.Omega_e, self.Omega_p),
            (self.omega_pe, self.omega_pp),
            self.omega,
            X_range,
        )

        values_dict = {
            "Omega": (self.Omega_e.value, self.Omega_p.value),
            "omega_p": (self.omega_pe.value, self.omega_pp.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(self.dispersion._poly_k, values_dict)

        dD_dw_sym = dispersion_poly_k.diff("omega")
        dD_dk_sym = dispersion_poly_k.diff("k")

        dD_dw = sym.lambdify(
            ["X", "k"],
            replace_cpdr_symbols(dD_dw_sym, {"omega": self.omega.value}),
            "numpy",
        )

        dD_dk = sym.lambdify(
            ["X", "k"],
            replace_cpdr_symbols(dD_dk_sym, {"omega": self.omega.value}),
            "numpy",
        )

        values_dict["omega"] = self.omega.value

        R_sym = replace_cpdr_symbols(self.dispersion._R, values_dict)
        assert np.isclose(float(R_sym), self.dispersion.stix.R(self.omega).value)
        assert np.isclose(float(R_sym), 1, rtol=1e3)

        L_sym = replace_cpdr_symbols(self.dispersion._L, values_dict)
        assert np.isclose(float(L_sym), self.dispersion.stix.L(self.omega).value)
        assert np.isclose(float(L_sym), 1, rtol=1e3)

        P_sym = replace_cpdr_symbols(self.dispersion._P, values_dict)
        assert np.isclose(float(P_sym), self.dispersion.stix.P(self.omega).value)
        assert np.isclose(float(P_sym), 1, rtol=1e3)

        S_sym = replace_cpdr_symbols(self.dispersion._S, values_dict)
        assert np.isclose(float(S_sym), self.dispersion.stix.S(self.omega).value)
        assert np.isclose(float(S_sym), 1, rtol=1e3)

        D_sym = replace_cpdr_symbols(self.dispersion._D, values_dict)
        assert np.isclose(float(D_sym), self.dispersion.stix.D(self.omega).value)
        assert np.isclose(float(D_sym), 1, rtol=1e3)

        dR_sym = self.dispersion._R.diff("omega")
        dR_sym = replace_cpdr_symbols(dR_sym, values_dict)
        assert np.isclose(float(dR_sym), self.dispersion.stix.dR(self.omega).value)
        assert np.isclose(float(dR_sym), 1, rtol=1e3)

        dL_sym = self.dispersion._L.diff("omega")
        dL_sym = replace_cpdr_symbols(dL_sym, values_dict)
        assert np.isclose(float(dL_sym), self.dispersion.stix.dL(self.omega).value)
        assert np.isclose(float(dL_sym), 1, rtol=1e3)

        dP_sym = self.dispersion._P.diff("omega")
        dP_sym = replace_cpdr_symbols(dP_sym, values_dict)
        assert np.isclose(float(dP_sym), self.dispersion.stix.dP(self.omega).value)
        assert np.isclose(float(dP_sym), 1, rtol=1e3)

        dS_sym = self.dispersion._S.diff("omega")
        dS_sym = replace_cpdr_symbols(dS_sym, values_dict)
        assert np.isclose(float(dS_sym), self.dispersion.stix.dS(self.omega).value)
        assert np.isclose(float(dS_sym), 1, rtol=1e3)

        dD_sym = self.dispersion._D.diff("omega")
        dD_sym = replace_cpdr_symbols(dD_sym, values_dict)
        assert np.isclose(float(dD_sym), self.dispersion.stix.dD(self.omega).value)
        assert np.isclose(float(dD_sym), 1, rtol=1e3)

        A_sym = replace_cpdr_symbols(self.dispersion._A, values_dict)
        A_ = sym.lambdify(["X"], A_sym, "numpy")

        B_sym = replace_cpdr_symbols(self.dispersion._B, values_dict)
        B_ = sym.lambdify(["X"], B_sym, "numpy")

        C_sym = replace_cpdr_symbols(self.dispersion._C, values_dict)
        C_ = sym.lambdify(["X"], C_sym, "numpy")

        dA_sym = replace_cpdr_symbols(self.dispersion._A.diff("omega"), values_dict)
        dA_ = sym.lambdify(["X"], dA_sym, "numpy")

        dB_sym = replace_cpdr_symbols(self.dispersion._B.diff("omega"), values_dict)
        dB_ = sym.lambdify(["X"], dB_sym, "numpy")

        dC_sym = replace_cpdr_symbols(self.dispersion._C.diff("omega"), values_dict)
        dC_ = sym.lambdify(["X"], dC_sym, "numpy")

        sympy_results = np.empty(len(xwk_roots), dtype=np.float64)
        numeric_results = np.empty(len(xwk_roots), dtype=np.float64)
        for ii, pair in enumerate(xwk_roots):
            X = pair[0]
            k = pair[2]

            # This fails because A_ from sympy route already contains mu...
            assert np.isclose(float(A_(X)), self.dispersion.stix.A(self.omega, X).value)
            assert np.isclose(float(B_(X)), self.dispersion.stix.B(self.omega, X).value)
            assert np.isclose(float(C_(X)), self.dispersion.stix.C(self.omega, X).value)

            assert np.isclose(
                float(dA_(X)), self.dispersion.stix.dA(self.omega, X).value
            )
            assert np.isclose(
                float(dB_(X)), self.dispersion.stix.dB(self.omega, X).value
            )
            assert np.isclose(
                float(dC_(X)), self.dispersion.stix.dC(self.omega, X).value
            )

            assert np.isclose(
                float(dD_dk(X, k)),
                self.dispersion.stix.dD_dk(self.omega, X, k / u.m).value,
            )
            assert np.isclose(
                float(dD_dw(X, k)),
                self.dispersion.stix.dD_dw(self.omega, X, k / u.m).value,
            )

            sympy_results[ii] = (k * dD_dw(X, k)) / ((1 + X**2) * dD_dk(X, k))

            numeric_results[ii] = self.dispersion.stix.jacobian(
                self.omega, X, k / u.m
            ).value

            print(
                f"{ii} : Sympy {sympy_results[ii]} : Numeric {numeric_results[ii]} : Ratio {sympy_results[ii]/numeric_results[ii]}"
            )

        # This assertion will pass, but *only just*
        assert np.allclose(sympy_results, numeric_results, atol=1e-20, equal_nan=False)
