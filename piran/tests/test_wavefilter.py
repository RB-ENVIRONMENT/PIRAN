import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.helpers import get_real_and_positive_roots
from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix
from piran.wavefilter import WaveFilter, WhistlerFilter


@u.quantity_input
def calculate_omega_L0(
    omega_p: u.Quantity[u.rad / u.s], omega_c: u.Quantity[u.rad / u.s]
) -> u.Quantity[u.rad / u.s]:
    """
    Whistler mode waves typically only 'overlap' with one other type of wave, Z-mode,
    which can occur when omega is such that L > 0.

    Let omega_L0 be the value of omega such that L = 0. By the defn of L this yields:

    1 - ((omega_p ** 2) / (omega_L0 * (omega_L0 - omega_c))) = 0

    NB. we're including the sign of the charge in omega_c here, which is negative
        for electrons.

    omega_c is fixed, so by inspection of the above both of the following are true:
    - increasing omega_p corresponds to increasing (unsigned) omega_L0.
    - omega_p > omega_L0.

    By defn, for an overdense plasma we have omega_p > abs(omega_c)
             for an underdense plasma we have omega_p < abs(omega_c)

    But how does omega_L0 compare to omega_c in different cases?

    Underdense plasma is easy: omega_L0 < omega_p < abs(omega_c).

    For an overdense plasma, there are two different scenarios. It is intuitive to let
    omega_L0 and omega_c coincide (i.e. omega_L0 = -omega_c), yielding

    1 - ((omega_p ** 2) / (2 * (omega_c ** 2))) = 0

    implying omega_p / abs(omega_c) = sqrt(2).

    For the overdense plasma, we thus have:

    omega_L0 < abs(omega_c) < omega_p if the plasma-over-gyro-ratio is < sqrt(2),
    abs(omega_c) < omega_L0 < omega_p otherwise.

    In "Introduction to Plasma Physics (Second Edition)" by Gurnett & Bhattacharjee,
    Section 4.4.4, Figures 4.37 and 4.38 plot the overdense case with ratio < sqrt(2)
    and the underdense case respectively.

    The remaining overdense case (ratio > sqrt(2)) is arguably more general!

    How do additional ions influence this analysis?...

    If we want to calculate omega_L0 directly given omega_p and omega_c, rearranging
    L=0 yields an easily-solved polynomial for omega_L0 as demonstrated in this
    function. Only the positive solution is retained.
    """
    L_quadratic = np.polynomial.Polynomial([-omega_p.value**2, -omega_c.value, 1])
    L0 = L_quadratic.roots()
    return L0[L0 > 0] << u.rad / u.s


class TestWaveFilter:
    def setup_method(self):
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5
        self.mag_point = MagPoint(mlat_deg, l_shell)
        self.particles = ("e", "p+")

    def test_abstract_filters(self):
        # Instantiate abstract WaveFilter
        with pytest.raises(TypeError):
            WaveFilter()

        # Define MissingFilter, inheriting from abstract WaveFilter,
        # _without_ implementing `filter` method
        class MissingFilter(WaveFilter):
            pass

        # Instantiate MissingFilter
        with pytest.raises(TypeError):
            MissingFilter()

    def test_bad_filter(self):
        # Define a BadFilter, where we have implemented the `filter` method
        # but its signature is inconsistent with our parent method.
        class BadFilter(WaveFilter):
            def filter(self):
                return "Hello world!"

        # Python allows this but I wish it didn't
        BadFilter()

        # And we can call `filter` without trouble:
        BadFilter().filter()

    def test_notimplemented_filter(self):
        # Define an InheritedFilter, where we have implemented the `filter` method
        # but it just calls the (NotImplemented) parent method.
        class InheritedFilter(WaveFilter):
            def filter(
                self,
                X: u.Quantity[u.dimensionless_unscaled],
                omega: u.Quantity[u.rad / u.s],
                k: u.Quantity[u.rad / u.m],
                plasma: PlasmaPoint,
                stix: Stix,
            ) -> u.Quantity[u.rad / u.m]:
                return super().filter(X, omega, k, plasma, stix)

        # We can instantiate this
        InheritedFilter()

        # But we can't use the inherited method
        # Params specified here don't matter...
        X = 0.3165829145728643 << u.dimensionless_unscaled
        omega = 21197.313961282573 << u.rad / u.s
        k = 0.00024206540583296198 << u.rad / u.m

        plasma_over_gyro_ratio = 1.5
        plasma = PlasmaPoint(self.mag_point, self.particles, plasma_over_gyro_ratio)
        stix = Stix(plasma.plasma_freq, plasma.gyro_freq)

        with pytest.raises(NotImplementedError):
            InheritedFilter().filter(X, omega, k, plasma, stix)

    def test_whistler_1(self):

        # Overdense plasma
        plasma_over_gyro_ratio = 1.5
        plasma = PlasmaPoint(self.mag_point, self.particles, plasma_over_gyro_ratio)
        stix = Stix(plasma.plasma_freq, plasma.gyro_freq)

        X = 0.3165829145728643 << u.dimensionless_unscaled
        omega = 21197.313961282573 << u.rad / u.s
        k = 0.00024206540583296198 << u.rad / u.m

        whistlers = WhistlerFilter()

        # Omega below omega_L0 (when Z-mode first appears)
        omega_p = plasma.plasma_freq[0]
        omega_c = plasma.gyro_freq[0]
        assert omega < calculate_omega_L0(omega_p, omega_c)

        # (omega, k) is whistler mode wave so filter should return True
        assert whistlers.filter(X, omega, k, plasma, stix)

    def test_whistler_2(self):

        # Underdense plasma
        plasma_over_gyro_ratio = 0.75
        plasma = PlasmaPoint(self.mag_point, self.particles, plasma_over_gyro_ratio)
        stix = Stix(plasma.plasma_freq, plasma.gyro_freq)

        energy = 1.0507018 * u.MeV
        alpha = Angle(0.125, u.deg)
        resonance = -5
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)
        cpdr = Cpdr(
            plasma,
            energy,
            alpha,
            resonance,
            freq_cutoff_params,
            WhistlerFilter(),
        )

        # We need an omega value larger than omega_L0 to be confident we are looking at both
        # whistler and z-mode waves
        omega_L0 = calculate_omega_L0(plasma.plasma_freq[0], plasma.gyro_freq[0])

        X = 0.259 << u.dimensionless_unscaled

        ### Method 1. ###
        ### Hardcoded omega (correspond to X above), bypassing cpdr.solve_resonant
        omega = 34357.65267505 << u.rad / u.s

        # Confirm we are above omega_L0 but below omega_p
        assert plasma.plasma_freq[0] > omega > omega_L0

        # Solve unmodified CPDR to obtain k roots for given X, omega
        k_l = cpdr.numpy_poly_in_k(X.value, omega.value).roots()

        # Keep only real and positive roots
        valid_k_l = get_real_and_positive_roots(k_l) << u.rad / u.m

        # Confirm we have 2 roots
        assert valid_k_l.size == 2

        is_desired_wave_mode = []
        for valid_k in valid_k_l:
            is_desired_wave_mode.append(
                WhistlerFilter().filter(X, omega, valid_k, plasma, stix)
            )

        # Confirm that we have 1 solution after filtering...
        k_after_filtering = valid_k_l[is_desired_wave_mode]
        assert k_after_filtering.size == 1

        # ... and that it is the largest solution (since index of refraction is greater for whistlers than z-mode)
        assert all(k_after_filtering >= valid_k_l)

        ### Method 2. ###
        ### Via cpdr.solve_resonant

        resonant_roots = cpdr.solve_resonant(X)

        # We only expect one soln for this X
        # (using prior knowledge)
        assert len(resonant_roots) == 1

        # Check that it is the same soln as found via method 1
        assert math.isclose(resonant_roots[0][0].k.value, k_after_filtering[0].value)
