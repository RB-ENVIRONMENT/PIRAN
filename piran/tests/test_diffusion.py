import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.diffusion import (
    get_diffusion_coefficients,
    get_DnX_single_root,
    get_energy_diffusion_coefficient,
    get_normalised_intensity,
    get_phi_squared,
    get_power_spectral_density,
    get_singular_term,
)
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint


class TestDiffusion:
    def setup_method(self):
        self.mlat_deg = Angle(0 * u.deg)
        self.l_shell = 4.5
        self.mag_point = MagPoint(self.mlat_deg, self.l_shell)
        self.particles = ("e", "p+")
        self.n_particles = len(self.particles)
        self.cpdr_sym = CpdrSymbolic(self.n_particles)
        self.freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

    def test_get_power_spectral_density_1(self):
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(
            self.mag_point, self.particles, plasma_over_gyro_ratio
        )
        energy = 1.0 << u.MeV
        alpha = Angle(70, u.deg)
        resonance = -1
        wave_amplitude = (100 << u.pT).to(u.T)

        cpdr = Cpdr(
            self.cpdr_sym,
            plasma_point,
            energy,
            alpha,
            resonance,
            self.freq_cutoff_params,
        )

        # Check the value and unit for one omega
        X = [0.5] << u.dimensionless_unscaled
        resonant_root = cpdr.solve_resonant(X)
        omega = resonant_root[0][0].omega
        power_spectral_density = get_power_spectral_density(cpdr, wave_amplitude, omega)

        assert omega.unit == u.rad / u.s
        assert math.isclose(omega.value, 8606.8, rel_tol=1e-5)

        assert power_spectral_density.unit == u.T**2 * u.s / u.rad
        assert math.isclose(power_spectral_density.value, 9.885e-26, rel_tol=1e-5)

        # If we integrate over the range [omega_lc, omega_uc]
        # the result should be B_w^2 (wave amplitude squared)
        omega_min = cpdr.omega_lc
        omega_max = cpdr.omega_uc
        omega_npoints = 100
        omega_range = u.Quantity(np.linspace(omega_min, omega_max, omega_npoints))
        eval_psd = []
        for omega_val in omega_range:
            eval_psd.append(get_power_spectral_density(cpdr, wave_amplitude, omega_val))
        eval_psd = u.Quantity(eval_psd, dtype=np.float64)
        integral = simpson(eval_psd, x=omega_range)
        assert math.isclose(integral, wave_amplitude.value**2, rel_tol=1e-7)

    def test_get_phi_squared_1(self):
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(
            self.mag_point, self.particles, plasma_over_gyro_ratio
        )
        energy = 1.0 << u.MeV
        alpha = Angle(70, u.deg)
        resonance = -1

        cpdr = Cpdr(
            self.cpdr_sym,
            plasma_point,
            energy,
            alpha,
            resonance,
            self.freq_cutoff_params,
        )

        X = [0.1, 0.5, 0.9] << u.dimensionless_unscaled
        resonant_root = cpdr.solve_resonant(X)

        phi_squared_1 = get_phi_squared(cpdr, resonant_root[0][0])  # neg k_par
        phi_squared_2 = get_phi_squared(cpdr, resonant_root[1][0])  # neg k_par
        phi_squared_3 = get_phi_squared(cpdr, resonant_root[2][0])  # neg k_par

        assert math.isclose(phi_squared_1, 0.496339, rel_tol=1e-6)
        assert math.isclose(phi_squared_2, 0.418072, rel_tol=1e-6)
        assert math.isclose(phi_squared_3, 0.283105, rel_tol=1e-6)

    def test_get_phi_squared_2(self):
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(
            self.mag_point, self.particles, plasma_over_gyro_ratio
        )
        energy = 1.0 << u.MeV
        alpha = Angle(83, u.deg)
        resonance = -1

        cpdr = Cpdr(
            self.cpdr_sym,
            plasma_point,
            energy,
            alpha,
            resonance,
            self.freq_cutoff_params,
        )

        X = [0.1] << u.dimensionless_unscaled
        resonant_root = cpdr.solve_resonant(X)

        phi_squared_11 = get_phi_squared(cpdr, resonant_root[0][0])  # pos k_par
        phi_squared_12 = get_phi_squared(cpdr, resonant_root[0][1])  # neg k_par

        assert math.isclose(phi_squared_11, 0.460906, rel_tol=1e-6)
        assert math.isclose(phi_squared_12, 0.489358, rel_tol=1e-6)

    def test_get_singular_term_1(self):
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(
            self.mag_point, self.particles, plasma_over_gyro_ratio
        )
        energy = 1.0 << u.MeV
        alpha = Angle(83, u.deg)
        resonance = -1

        cpdr = Cpdr(
            self.cpdr_sym,
            plasma_point,
            energy,
            alpha,
            resonance,
            self.freq_cutoff_params,
        )

        X = [0.1] << u.dimensionless_unscaled
        resonant_root = cpdr.solve_resonant(X)

        # positive k_par
        singular_term_11 = get_singular_term(cpdr, resonant_root[0][0])
        assert singular_term_11.unit == u.m / u.s
        assert math.isclose(singular_term_11.value, -54012493.8, rel_tol=1e-7)

        # negative k_par
        singular_term_12 = get_singular_term(cpdr, resonant_root[0][1])
        assert singular_term_12.unit == u.m / u.s
        assert math.isclose(singular_term_12.value, 154355842.6, rel_tol=1e-7)

    def test_get_normalised_intensity_1(self):
        power_spectral_density = 1.5079984e-25 << (u.T**2 * u.s / u.rad)
        gx = 0.97041
        norm_factor = 1.754757e-17

        normalised_intensity = get_normalised_intensity(
            power_spectral_density, gx, norm_factor
        )
        assert math.isclose(normalised_intensity.value, 8.339484e-09, rel_tol=1e-7)

    def test_get_DnX_single_root(self):
        plasma_over_gyro_ratio = 1.5
        plasma_point = PlasmaPoint(
            self.mag_point, self.particles, plasma_over_gyro_ratio
        )
        energy = 1.0 << u.MeV
        alpha = Angle(83, u.deg)
        resonance = -1

        cpdr = Cpdr(
            self.cpdr_sym,
            plasma_point,
            energy,
            alpha,
            resonance,
            self.freq_cutoff_params,
        )

        X = [0.1] << u.dimensionless_unscaled
        resonant_root = cpdr.solve_resonant(X)

        normalised_intensity = 8.339484e-09
        phi_squared = 0.460906 << u.dimensionless_unscaled
        singular_term = -54012493.87 << (u.m / u.s)

        DnXaa, DnXap, DnXpp = get_DnX_single_root(
            cpdr,
            resonant_root[0][0],
            normalised_intensity,
            phi_squared,
            singular_term,
        )

        assert math.isclose(DnXaa.value, 1.1892186e-45, rel_tol=1e-7)
        assert math.isclose(DnXap.value, -4.1240996e-46, rel_tol=1e-7)
        assert math.isclose(DnXpp.value, 1.4301994e-46, rel_tol=1e-7)

    def test_get_diffusion_coefficients(self):
        X = [0, 0.2, 0.4, 0.6, 0.8, 1.0] << u.dimensionless_unscaled
        DnX = np.array([0, 1, 2, 3, 4, 5])
        res = get_diffusion_coefficients(X, DnX)

        assert math.isclose(res, 1.666, rel_tol=1e-3)

        # raise error when X and DnX do not have the same shape
        X = [0, 0.2, 0.4, 0.6, 0.8, 1.0] << u.dimensionless_unscaled
        DnX = np.array([0, 1, 2, 3, 4])

        with pytest.raises(ValueError):
            get_diffusion_coefficients(X, DnX)

    def test_get_energy_diffusion_coefficient(self):
        rel_kin_energy = 1.602177e-15 << u.J
        rest_mass_energy = 8.187106e-14 << u.J
        momentum_diff_coef = 7.073597e-50

        Dee = get_energy_diffusion_coefficient(
            rel_kin_energy,
            rest_mass_energy,
            momentum_diff_coef,
        )

        assert math.isclose(Dee, 2.41705e-34, rel_tol=1e-5)
