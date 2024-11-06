import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.bounce import Bounce
from piran.magpoint import MagPoint


class TestBounce:
    def test_bounce_1(self):
        """Check that equatorial pitch angle is in (0, 90) degrees."""

        # mlat in degrees
        mlat = 0.0 << u.rad
        l_shell = 4.5
        eq_mag_point = MagPoint(mlat, l_shell)

        # Pitch angle 0.0
        eq_pitch_angle = 0.0 << u.rad
        with pytest.raises(ValueError):
            Bounce(eq_pitch_angle, eq_mag_point)

        # Pitch angle 90.0 degrees
        eq_pitch_angle = np.pi / 2 << u.rad
        with pytest.raises(ValueError):
            Bounce(eq_pitch_angle, eq_mag_point)

    def test_bounce_2(self):
        """Check that equatorial magnetic latitude is 0."""

        # mlat in degrees
        mlat_deg = Angle(1, u.deg)
        l_shell = 4.5
        eq_mag_point = MagPoint(mlat_deg, l_shell)

        eq_pitch_angle = np.pi / 4 << u.rad
        with pytest.raises(ValueError):
            Bounce(eq_pitch_angle, eq_mag_point)

        # mlat in radians
        mlat_rad = Angle(0.1, u.rad)
        l_shell = 4.5
        eq_mag_point = MagPoint(mlat_rad, l_shell)

        eq_pitch_angle = np.pi / 4 << u.rad
        with pytest.raises(ValueError):
            Bounce(eq_pitch_angle, eq_mag_point)

    def test_bounce_3(self):
        """Test the `get_particle_bounce_period` method."""
        mlat = Angle(0.0, u.rad)
        l_shell = 4.5
        mag_point_eq = MagPoint(mlat, l_shell)

        # Equatorial pitch angle: pi/1000 rad
        a_eq = np.pi / 1000 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.particle_bounce_period, 1.29824, rel_tol=1e-05)

        # Equatorial pitch angle: pi/8 rad
        a_eq = np.pi / 8 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.particle_bounce_period, 1.08569, rel_tol=1e-05)

        # Equatorial pitch angle: pi/4 rad
        a_eq = np.pi / 4 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.particle_bounce_period, 0.90402, rel_tol=1e-05)

        # Equatorial pitch angle: pi/2 - 0.0001 rad
        a_eq = np.pi / 2 - 0.0001 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.particle_bounce_period, 0.74, rel_tol=1e-05)

    def test_bounce_4(self):
        """Test the `get_mirror_latitude` method."""
        mlat = Angle(0.0, u.rad)
        l_shell = 4.5
        mag_point_eq = MagPoint(mlat, l_shell)

        # Equatorial pitch angle: pi/1000 rad
        a_eq = np.pi / 1000 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.mirror_latitude.value, 1.405935, rel_tol=1e-05)

        # Equatorial pitch angle: pi/8 rad
        a_eq = np.pi / 8 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.mirror_latitude.value, 0.68371, rel_tol=1e-05)

        # Equatorial pitch angle: pi/4 rad
        a_eq = np.pi / 4 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.mirror_latitude.value, 0.403735, rel_tol=1e-05)

        # Equatorial pitch angle: pi/2 - 0.0001 rad
        a_eq = np.pi / 2 - 0.0001 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        assert math.isclose(bounce.mirror_latitude.value, 4.714e-05, rel_tol=1e-05)

    def test_bounce_5(self):
        """Test the `get_bounce_pitch_angle` method."""
        mag_point_eq = MagPoint(0.0 << u.rad, 4.5)

        # Equatorial pitch angle: pi/1000 rad
        a_eq = np.pi / 1000 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        pitch_angle_1 = bounce.get_bounce_pitch_angle(0.0 << u.rad)
        pitch_angle_2 = bounce.get_bounce_pitch_angle(
            bounce.mirror_latitude - (0.00000001 << u.rad)
        )
        pitch_angle_3 = bounce.get_bounce_pitch_angle(bounce.mirror_latitude)
        assert math.isclose(pitch_angle_1.value, 0.00314159, rel_tol=1e-05)
        assert math.isclose(pitch_angle_2.value, 1.57019476, rel_tol=1e-05)
        # Temporarily commented out as the result is too sensitive to NumPy
        # version differences (NaN or pi/2).
        # assert np.isnan(pitch_angle_3.value)

        # Equatorial pitch angle: pi/4 rad
        a_eq = np.pi / 4 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        pitch_angle_1 = bounce.get_bounce_pitch_angle(0.0 << u.rad)
        pitch_angle_2 = bounce.get_bounce_pitch_angle(
            bounce.mirror_latitude - (0.00000001 << u.rad)
        )
        pitch_angle_3 = bounce.get_bounce_pitch_angle(bounce.mirror_latitude)
        assert math.isclose(pitch_angle_1.value, 0.78539816, rel_tol=1e-05)
        assert math.isclose(pitch_angle_2.value, 1.57061455, rel_tol=1e-05)
        assert np.isnan(pitch_angle_3.value)

        # Equatorial pitch angle: pi/2 - 0.0001 rad
        a_eq = np.pi / 2 - 0.0001 << u.rad
        bounce = Bounce(a_eq, mag_point_eq)
        pitch_angle_1 = bounce.get_bounce_pitch_angle(0.0 << u.rad)
        pitch_angle_2 = bounce.get_bounce_pitch_angle(
            bounce.mirror_latitude - (0.00000001 << u.rad)
        )
        pitch_angle_3 = bounce.get_bounce_pitch_angle(bounce.mirror_latitude)
        assert math.isclose(pitch_angle_1.value, 1.57069632, rel_tol=1e-05)
        assert math.isclose(pitch_angle_2.value, 1.57079426, rel_tol=1e-05)
        assert math.isclose(pitch_angle_3.value, np.pi / 2, rel_tol=1e-05)
