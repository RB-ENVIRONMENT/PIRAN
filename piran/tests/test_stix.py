import math

import numpy as np
import sympy as sym
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.gauss import Gaussian
from piran.magfield import MagField
from piran.normalisation import solve_dispersion_relation
from piran.particles import Particles, PiranParticle
from piran.resonance import replace_cpdr_symbols


class TestStix:
    def setup_method(self):
        frequency_ratio = 1.5 * u.dimensionless_unscaled
        omega_ratio = 0.1225

        # ============================== START ============================== #
        # Those should be attributes of one of the main classes
        # The *only* parts of these we still need are:
        # - mlat
        # - l_shell
        # - n_
        #
        # Unfortunately n_ requires everything else :(
        # Can we define something for PiranParticle that lets us just pass in a
        # frequency_ratio instead of a number density or plasma frequency?...
        # Uncertain - we would need access to the gyrofrequency which is currently
        # stored in the Cpdr class.

        M = 8.033454e15 * (u.tesla * u.m**3)
        mlat = Angle(0, u.deg)
        l_shell = 4.5 * u.dimensionless_unscaled
        B = (M * math.sqrt(1 + 3 * math.sin(mlat.rad) ** 2)) / (
            l_shell**3 * const.R_earth**3 * math.cos(mlat.rad) ** 6
        )

        q_e = -const.e.si  # Signed electron charge

        self.Omega_e = (q_e * B) / const.m_e
        Omega_e_abs = abs(self.Omega_e)
        self.omega_pe = Omega_e_abs * frequency_ratio

        n_ = self.omega_pe**2 * const.eps0 * const.m_e / abs(q_e) ** 2
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

        # Use "p+" for proton here instead of "H+".
        # "H+" accounts for hydrogen isotopes so has a higher standard atomic weight!
        piran_particle_list = (PiranParticle("e", n_), PiranParticle("p+", n_))
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

        self.omega = abs(self.dispersion._w_c[0]) * omega_ratio

        self.dispersion.as_poly_in_k()

    def test_jacobian(self):
        X_min = 0.00
        X_max = 1.00
        X_npoints = 100
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )

        # Find resonant (X, w, k) triplets
        xwk_roots = solve_dispersion_relation(
            self.dispersion,
            self.dispersion._w_c,
            self.dispersion._w_p,
            self.omega,
            X_range,
        )

        # Sub in known gyro- and plasma-frequencies before differentiating CPDR
        # (to give Sympy an easier time)
        values_dict = {
            "Omega": tuple(self.dispersion._w_c.value),
            "omega_p": tuple(self.dispersion._w_p.value),
        }
        dispersion_poly_k = replace_cpdr_symbols(self.dispersion._poly_k, values_dict)

        # Differentiate CPDR w.r.t. omega, k
        dD_dw_sym = dispersion_poly_k.diff("omega")
        dD_dk_sym = dispersion_poly_k.diff("k")

        # Sub known value for omega back in (now that we are past the
        # 'differentiate w.r.t. omega' step) and lambdify for speed.
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

        # Compare 'numeric' results versus similar 'sympy' results for all resonant
        # (X, w, k) triplets
        for pair in xwk_roots:
            X = pair[0]
            k = pair[2]

            sympy_result = (k * dD_dw(X, k)) / ((1 + X**2) * dD_dk(X, k))

            numeric_result = self.dispersion.stix.jacobian(self.omega, X, k / u.m).value

            assert math.isclose(sympy_result, numeric_result)
