import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.cpdr import Cpdr
from piran.cpdrsymbolic import CpdrSymbolic
from piran.magpoint import MagPoint
from piran.meshing import count_roots_per_bucket, solve_resonant_for_x, split_array
from piran.plasmapoint import PlasmaPoint


class TestMeshing:

    def test_meshing_regular_case(self):
        # ================ Parameters =====================
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5

        energy = 1.0 * u.MeV
        alpha = Angle(65, u.deg)
        resonance = -4
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

        X_min = 0.0
        X_max = 1
        X_npoints = 1001
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )
        # =================================================

        mag_point = MagPoint(mlat_deg, l_shell)
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
        cpdr_sym = CpdrSymbolic(len(particles))
        cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
        )

        X_all = solve_resonant_for_x(
            cpdr, u.Quantity([cpdr.omega_uc, cpdr.omega_lc]), X_range, True
        )

        assert len(X_all) == 3
        assert math.isclose(X_all[0].value, 0)
        assert math.isclose(X_all[1].value, 0.4970500912721371)
        assert math.isclose(X_all[2].value, 1)

        buckets = split_array(X_all)

        assert len(buckets) == 2
        assert math.isclose(buckets[0][0], 0)
        assert math.isclose(buckets[0][1], 0.4970500912721371)
        assert math.isclose(buckets[1][0], 0.4970500912721371)
        assert math.isclose(buckets[1][1], 1)

        # Count roots in each bucket
        num_roots = count_roots_per_bucket(cpdr, buckets)

        assert len(num_roots) == 2
        assert num_roots[0] == 0
        assert num_roots[1] == 1

    def test_meshing_singular_case(self):
        # ================ Parameters =====================
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5

        energy = 1.0 * u.MeV
        alpha = Angle(70.8, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

        X_min = 0.0
        X_max = 1
        X_npoints = 1001
        X_range = u.Quantity(
            np.linspace(X_min, X_max, X_npoints), unit=u.dimensionless_unscaled
        )
        # =================================================

        mag_point = MagPoint(mlat_deg, l_shell)
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
        cpdr_sym = CpdrSymbolic(len(particles))
        cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
        )

        X_l = solve_resonant_for_x(cpdr, cpdr.omega_lc, X_range)
        X_u = solve_resonant_for_x(cpdr, cpdr.omega_uc, X_range)
        X_all = solve_resonant_for_x(
            cpdr, u.Quantity([cpdr.omega_uc, cpdr.omega_lc]), X_range
        )

        # 0 solutions corresponding to omega_lc
        assert len(X_l) == 0

        # 1 solution corresponding to omega_uc
        # Let's check that we see this for both omega_lc and (omega_lc, omega_uc) inputs
        assert len(X_u) == len(X_all) == 1
        assert math.isclose(X_u[0], X_all[0])
        assert math.isclose(X_u[0], 0.23727364529766135)

        # Grab X_all again, including endpoints X_min, X_max this time
        X_all_with_endpoints = solve_resonant_for_x(
            cpdr, u.Quantity([cpdr.omega_uc, cpdr.omega_lc]), X_range, True
        )

        # Check length is as expected
        assert len(X_all_with_endpoints) == len(X_all) + 2 == 3

        # Check value is as before
        # Index will be 1 since this is our 1 soln in X between X_min (index 0)
        # and X_max (index 2)
        assert math.isclose(X_all_with_endpoints[1], 0.23727364529766135)

        # Split the array X_all_with_endpoints into a list of distinct buckets
        buckets = split_array(X_all_with_endpoints)

        # Check length
        assert len(buckets) == len(X_all_with_endpoints) - 1 == 2

        # Count roots in each bucket
        num_roots = count_roots_per_bucket(cpdr, buckets)

        assert len(num_roots) == len(buckets) == 2
        assert num_roots[0] == 1
        assert np.isnan(num_roots[1])  # singularity detected!

        # There's a singularity _somewhere_ in our second bucket between 0.23727 and 1.
        # Lets try truncating our second bucket at X = 0.5 and see if this removes the
        # singularity.
        # NOTE: This is NOT expected usage of this API - I just want to test a 'regular'
        # bucket with 2 roots too and don't have another example to hand!

        buckets[1][1] = 0.5 << u.dimensionless_unscaled
        num_roots = count_roots_per_bucket(cpdr, buckets)
        assert num_roots[1] == 2

    def test_split_array_with_insufficient_values(self):
        """
        Check that split_array raises an exception when the input array contains fewer
        than 3 values.
        """

        array0 = u.Quantity([], u.dimensionless_unscaled)
        array1 = u.Quantity([1], u.dimensionless_unscaled)
        array2 = u.Quantity([1, 2], u.dimensionless_unscaled)

        with pytest.raises(ValueError):
            split_array(array0)

        with pytest.raises(ValueError):
            split_array(array1)

        with pytest.raises(ValueError):
            split_array(array2)

    def test_count_roots_per_bucket_with_unordered_buckets(self):
        # ================ Parameters =====================
        # We need a Cpdr when calling count_roots_per_bucket, but we're expecting the
        # method to raise an Exception before we ever use the Cpdr in these tests...
        # Parameters below don't really matter!

        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5

        particles = ("e", "p+")
        plasma_over_gyro_ratio = 1.5

        energy = 1.0 * u.MeV
        alpha = Angle(45, u.deg)
        resonance = 0
        freq_cutoff_params = (0.35, 0.15, -1.5, 1.5)

        mag_point = MagPoint(mlat_deg, l_shell)
        plasma_point = PlasmaPoint(mag_point, particles, plasma_over_gyro_ratio)
        cpdr_sym = CpdrSymbolic(len(particles))

        cpdr = Cpdr(
            cpdr_sym, plasma_point, energy, alpha, resonance, freq_cutoff_params
        )
        # =================================================

        bucket1 = u.Quantity([2, 1], u.dimensionless_unscaled)

        # This test should raise an exception because the ordering of elements in the
        # bucket [2, 1] is wrong
        with pytest.raises(ValueError):
            count_roots_per_bucket(cpdr, [bucket1])

        bucket2 = u.Quantity([0.5, 0.5001], u.dimensionless_unscaled)

        # This test should raise an exception because the elements in the bucket
        # [0.5, 0.5001] are sufficiently close together, and our eps=1e-4 is sufficiently
        # large, that the ordering of sample points 'near' each endpoints overlaps.
        # This probably highlights that we should try implementing something better in
        # this function...
        with pytest.raises(ValueError):
            count_roots_per_bucket(cpdr, [bucket2], 1e-4)

        # This should succeed without raising an Exception, although we don't care about
        # the results here otherwise.
        count_roots_per_bucket(cpdr, [bucket2], 1e-8)
