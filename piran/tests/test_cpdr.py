import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr2 import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
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

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        energy = 1.0 * u.MeV
        alpha = Angle(5, u.deg)
        resonance = 2
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        self.cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
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
        assert math.isclose(k[0], 0.00011414445445389277)
        assert math.isclose(k[1], 0.00011766325510931447)
        assert math.isclose(k[2], 0.00014032247090573543)
        assert math.isnan(k[3]) is True

        # Raise valuer error (more than 1 real and positive root k)
        omega = 100000 << (u.rad / u.s)
        X = [1.0] << u.dimensionless_unscaled
        with pytest.raises(ValueError):
            self.cpdr.solve_cpdr_for_norm_factor(omega, X)

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

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        energy = 1.0 * u.MeV
        alpha = Angle(71, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
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
        assert math.isclose(roots[1][0].omega.value, 34361.48787566025)
        assert math.isclose(roots[1][0].k.value, 0.0003923953536206822)
        assert math.isclose(roots[1][1].X.value, 0.3165829145728643)
        assert math.isclose(roots[1][1].omega.value, 21197.313961282573)
        assert math.isclose(roots[1][1].k.value, 0.00024206540583296198)

    def test_cpdr_5(self):
        """No resonant frequency"""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        energy = 1.0 * u.MeV
        alpha = Angle(70, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
        )

        X = [0.0] << u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert len(roots) == 1

        assert len(roots[0]) == 1
        assert math.isclose(roots[0][0].X.value, 0.0)
        assert np.isnan(roots[0][0].omega)
        assert np.isnan(roots[0][0].k)

    def test_cpdr_6(self):
        """Test k_par positive and negative"""
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        mag_point = MagPoint(mlat_deg, l_shell)

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)

        n_particles = len(particles)
        cpdr_sym = CpdrSymbolic(n_particles)

        energy = 1.0 * u.MeV
        alpha = Angle(83, u.deg)
        resonance = -1
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
        )

        X = [0.6] << u.dimensionless_unscaled
        roots = cpdr.solve_resonant(X)

        assert len(roots) == 1
        assert len(roots[0]) == 2

        assert math.isclose(roots[0][0].X.value, 0.6)
        assert math.isclose(roots[0][0].omega.value, 33719.09383836)
        assert math.isclose(roots[0][0].k.value, 0.0004581964569)
        assert math.isclose(roots[0][0].k_par.value, 0.0003929002204)
        assert math.isclose(roots[0][0].k_perp.value, 0.0002357401322)

        assert math.isclose(roots[0][1].X.value, 0.6)
        assert math.isclose(roots[0][1].omega.value, 14447.39075335)
        assert math.isclose(roots[0][1].k.value, 0.0001954579422)
        assert math.isclose(roots[0][1].k_par.value, -0.0001676038027)
        assert math.isclose(roots[0][1].k_perp.value, 0.0001005622816)
