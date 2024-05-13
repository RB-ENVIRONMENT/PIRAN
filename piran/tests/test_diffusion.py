import math

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from scipy.integrate import simpson

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.diffusion import get_phi_squared, get_power_spectral_density
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

        phi_squared_1 = get_phi_squared(cpdr, resonant_root[0][0])
        phi_squared_2 = get_phi_squared(cpdr, resonant_root[1][0])
        phi_squared_3 = get_phi_squared(cpdr, resonant_root[2][0])

        assert math.isclose(phi_squared_1, 0.497342, rel_tol=1e-6)
        assert math.isclose(phi_squared_2, 0.440568, rel_tol=1e-6)
        assert math.isclose(phi_squared_3, 0.340022, rel_tol=1e-6)
