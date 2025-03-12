# Copyright (C) 2025 The University of Birmingham, United Kingdom /
#   Dr Oliver Allanson, ORCiD: 0000-0003-2353-8586, School Of Engineering, University of Birmingham /
#   Dr Thomas Kappas, ORCiD: 0009-0003-5888-2093, Advanced Research Computing, University of Birmingham /
#   Dr James Tyrrell, ORCiD: 0000-0002-2344-737X, Advanced Research Computing, University of Birmingham /
#   Dr Adrian Garcia, ORCiD: 0009-0007-4450-324X, Advanced Research Computing, University of Birmingham
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import math

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


class TestCpdr:
    def setup_method(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        energy = 1.0 * u.MeV
        alpha = Angle(5, u.deg)
        resonance = 2
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        self.cpdr = Cpdr(
            plasma_point,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
        )

    def test_cpdr_1(self):
        assert math.isclose(self.cpdr.energy.value, 1.6021766339e-13)  # Joule
        assert math.isclose(self.cpdr.pitch_angle.value, 0.08726646259)  # radians
        assert math.isclose(self.cpdr.alpha.value, 0.08726646259)  # radians
        assert self.cpdr.resonance == 2
        assert math.isclose(self.cpdr.lorentz_factor, 2.956951183)
        assert math.isclose(self.cpdr.gamma, 2.956951183)
        assert math.isclose(self.cpdr.rel_velocity.value, 2.82128455e08)  # m/s
        assert math.isclose(self.cpdr.v_par.value, 2.81054871e08)  # m/s
        assert math.isclose(self.cpdr.omega_lc.value, 7470.0276, rel_tol=1e-6)  # rad/s
        assert math.isclose(self.cpdr.omega_uc.value, 34362.127, rel_tol=1e-6)  # rad/s
        assert math.isclose(self.cpdr.momentum.value, 7.5994128855e-22)  # kg*m/s
        assert math.isclose(self.cpdr.p_par.value, 7.5704948251e-22)  # kg*m/s
        assert math.isclose(self.cpdr.p_perp.value, 6.6233247448e-23)  # kg*m/s

        assert self.cpdr.energy.unit == u.Joule
        assert self.cpdr.pitch_angle.unit == u.rad
        assert self.cpdr.alpha.unit == u.rad
        assert self.cpdr.rel_velocity.unit == u.m / u.s
        assert self.cpdr.v_par.unit == u.m / u.s
        assert self.cpdr.omega_lc.unit == u.rad / u.s
        assert self.cpdr.omega_uc.unit == u.rad / u.s
        assert self.cpdr.momentum.unit == u.kg * u.m / u.s
        assert self.cpdr.p_par.unit == u.kg * u.m / u.s
        assert self.cpdr.p_perp.unit == u.kg * u.m / u.s

        # Eval gaussian at mean
        wave_freqs_eval1 = self.cpdr.wave_freqs.eval(20916.08 * (u.rad / u.s))
        assert math.isclose(wave_freqs_eval1.value, 1.0)

        # Eval above lower cutoff and below mean
        wave_freqs_eval2 = self.cpdr.wave_freqs.eval(10000 * (u.rad / u.s))
        assert math.isclose(wave_freqs_eval2.value, 0.2269673064)

        # Eval below lower cutoff
        wave_freqs_eval3 = self.cpdr.wave_freqs.eval(7000 * (u.rad / u.s))
        assert math.isclose(wave_freqs_eval3.value, 0.0)

        # Eval above upper cutoff
        wave_freqs_eval4 = self.cpdr.wave_freqs.eval(35000 * (u.rad / u.s))
        assert math.isclose(wave_freqs_eval4.value, 0.0)

    def test_cpdr_2(self):
        omega = 7320.627086050828 * (u.rad / u.s)

        X = [0.0, 0.33333333333333337, 1.0, 100] * u.dimensionless_unscaled
        k = self.cpdr.solve_cpdr_for_norm_factor(omega, X)
        assert math.isclose(k[0].value, 0.00011414445445389277)
        assert math.isclose(k[1].value, 0.00011766325510931447)
        assert math.isclose(k[2].value, 0.00014032247090573543)
        assert math.isnan(k[3].value)

        # Test whether we get [np.nan], as frequency omega is not within range
        # (value returned by the wave filter).
        omega = 100000 << (u.rad / u.s)
        X = [1.0] << u.dimensionless_unscaled
        k = self.cpdr.solve_cpdr_for_norm_factor(omega, X)
        assert np.isnan(k[0])  # same as math.isnan(k[0].value)

    def test_cpdr_3(self):
        X = [0.01, 0.99] * u.dimensionless_unscaled
        roots = self.cpdr.solve_resonant(X)

        assert len(roots) == 2
        assert len(roots[0]) == 1
        assert len(roots[1]) == 1

        assert math.isclose(roots[0][0].X.value, 0.01)
        assert math.isclose(roots[0][0].omega.value, 18549.99508102283)
        assert math.isclose(roots[0][0].k.value, 0.0002098277253605769)
        assert math.isclose(roots[0][0].k_par.value, 0.0002098172347610972)
        assert math.isclose(roots[0][0].k_perp.value, 0.0000020981723476109)

        assert math.isclose(roots[1][0].X.value, 0.99)
        assert math.isclose(roots[1][0].omega.value, 19814.68720788155)
        assert math.isclose(roots[1][0].k.value, 0.0003015784216619821)
        assert math.isclose(roots[1][0].k_par.value, 0.0002143170398076355)
        assert math.isclose(roots[1][0].k_perp.value, 0.0002121738694095591)

        assert roots[0][0].X.unit == u.dimensionless_unscaled
        assert roots[0][0].omega.unit == u.rad / u.s
        assert roots[0][0].k.unit == u.rad / u.m
        assert roots[0][0].k_par.unit == u.rad / u.m
        assert roots[0][0].k_perp.unit == u.rad / u.m

    def test_cpdr_4(self):
        """Different pitch angle alpha"""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        energy = 1.0 * u.MeV
        alpha = Angle(71, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            plasma_point,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
        )

        X = [0.0, 0.3165829145728643] * u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert len(roots) == 2

        assert len(roots[0]) == 1
        assert math.isclose(roots[0][0].X.value, 0.0)
        assert math.isclose(roots[0][0].omega.value, 22060.04543112965)
        assert math.isclose(roots[0][0].k.value, 0.00024016935645729707)

        assert len(roots[1]) == 2
        assert math.isclose(roots[1][0].X.value, 0.3165829145728643)
        assert math.isclose(roots[1][0].omega.value, 21197.313961282573)
        assert math.isclose(roots[1][0].k.value, 0.00024206540583296198)
        assert math.isclose(roots[1][1].X.value, 0.3165829145728643)
        assert math.isclose(roots[1][1].omega.value, 34361.48787566025)
        assert math.isclose(roots[1][1].k.value, 0.0003923953536206822)

    def test_cpdr_5(self):
        """No resonant frequency"""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        energy = 1.0 * u.MeV
        alpha = Angle(70, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            plasma_point,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
        )

        X = [0.0] << u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert len(roots) == 1

        assert len(roots[0]) == 1
        assert math.isclose(roots[0][0].X.value, 0.0)
        assert np.isnan(roots[0][0].omega)
        assert np.isnan(roots[0][0].k)

    def test_cpdr_6(self):
        """More solve_resonant and find_resonant_wavenumber tests"""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        energy = 1.0 * u.MeV
        alpha = Angle(83, u.deg)
        resonance = -1
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            plasma_point,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
        )

        # Test k_par positive and negative
        X = [0.6] << u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert len(roots) == 1
        assert len(roots[0]) == 2

        assert math.isclose(roots[0][0].X.value, 0.6)
        assert math.isclose(roots[0][0].omega.value, 14447.39075335)
        assert math.isclose(roots[0][0].k.value, 0.0001954579422)
        assert math.isclose(roots[0][0].k_par.value, -0.0001676038027)
        assert math.isclose(roots[0][0].k_perp.value, 0.0001005622816)

        assert math.isclose(roots[0][1].X.value, 0.6)
        assert math.isclose(roots[0][1].omega.value, 33719.09383836)
        assert math.isclose(roots[0][1].k.value, 0.0004581964569)
        assert math.isclose(roots[0][1].k_par.value, 0.0003929002204)
        assert math.isclose(roots[0][1].k_perp.value, 0.0002357401322)

    def test_find_resonant_wavenumber_1(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        energy = 1.0 << u.MeV
        alpha = Angle(89.5, u.deg)
        resonance = -1
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            plasma_point,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
        )

        X = 0.0 << u.dimensionless_unscaled
        omega = 20773.61527263705 << u.rad / u.s

        k, k_par, k_perp = cpdr.find_resonant_wavenumber(X, omega)
        assert math.isclose(k.value, 0.00022889276, rel_tol=1e-08)
        assert math.isclose(k_par.value, 0.00022889276, rel_tol=1e-08)
        assert math.isclose(k_perp.value, 0.0)

    def test_filtering(self):
        """Test where (omega, k) is not in the desired wave mode."""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 0.2
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        energy = 0.1 * u.MeV
        alpha = Angle(5, u.deg)
        resonance = -3
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            plasma_point,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
        )

        X = [0.0] << u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert math.isclose(roots[0][0].omega.value, 11955.2984, rel_tol=1e-08)
        assert np.isnan(roots[0][0].k)
        assert np.isnan(roots[0][0].k_par)
        assert np.isnan(roots[0][0].k_perp)
