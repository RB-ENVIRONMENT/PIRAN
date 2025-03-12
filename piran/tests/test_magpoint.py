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

import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.magpoint import MagPoint


class TestMagPoint:
    def test_magpoint_1(self):
        """
        For default planet (Earth).
        Check also if an error is raised when trying to
        modify variables, as there are no setters.
        """
        mlat_deg = Angle(0 * u.deg)
        l_shell = 4.5 * u.dimensionless_unscaled

        mag_point = MagPoint(mlat_deg, l_shell)

        assert mag_point.mlat.deg == 0  # deg
        assert mag_point.mlat == 0  # rad
        assert mag_point.l_shell == 4.5
        assert mag_point.planetary_radius.value == 6378100.0  # meters
        assert mag_point.mag_dipole_moment.value == 8.033454e15  # T*m^3
        assert math.isclose(mag_point.flux_density.value, 3.39774512e-07)  # T

        # We don't have setters, only getters
        with pytest.raises(AttributeError):
            mag_point.mlat = Angle(10 * u.deg)

        with pytest.raises(AttributeError):
            mag_point.l_shell = 5.0 * u.dimensionless_unscaled

        with pytest.raises(AttributeError):
            mag_point.planetary_radius = 100 * u.km

        with pytest.raises(AttributeError):
            mag_point.mag_dipole_moment = 1 * (u.tesla * u.m**3)

        with pytest.raises(AttributeError):
            mag_point.flux_density = 1 * u.tesla

    def test_magpoint_2(self):
        mlat_deg = Angle(10 * u.deg)
        l_shell = 5.0 * u.dimensionless_unscaled

        mag_point = MagPoint(mlat_deg, l_shell)

        assert math.isclose(mag_point.mlat.deg, 10, rel_tol=1e-3)  # deg
        assert math.isclose(mag_point.mlat.value, 0.174532925, rel_tol=1e-8)  # rad
        assert mag_point.l_shell == 5.0
        assert mag_point.planetary_radius.value == 6378100.0  # meters
        assert mag_point.mag_dipole_moment.value == 8.033454e15  # T*m^3
        assert math.isclose(mag_point.flux_density.value, 2.835402086e-07)  # T

    def test_magpoint_3(self):
        mlat_deg = Angle(20 * u.deg)
        l_shell = 5.0 * u.dimensionless_unscaled
        mag_dipole_moment = 7e15 * (u.tesla * u.m**3)

        mag_point = MagPoint(
            mlat_deg,
            l_shell,
            mag_dipole_moment=mag_dipole_moment,
        )

        assert math.isclose(mag_point.mlat.deg, 20, rel_tol=1e-3)  # deg
        assert math.isclose(mag_point.mlat.value, 0.349065852, rel_tol=1e-8)  # rad
        assert mag_point.l_shell == 5.0
        assert mag_point.planetary_radius.value == 6378100.0  # meters
        assert mag_point.mag_dipole_moment.value == 7e15  # T*m^3
        assert math.isclose(mag_point.flux_density.value, 3.643477695e-07)  # T

    def test_magpoint_4(self):
        mlat_deg = Angle(0.349065852 * u.rad)
        l_shell = 5.0 * u.dimensionless_unscaled
        planetary_radius = 6000 * u.km
        mag_dipole_moment = 7e15 * (u.tesla * u.m**3)

        mag_point = MagPoint(
            mlat_deg,
            l_shell,
            planetary_radius=planetary_radius,
            mag_dipole_moment=mag_dipole_moment,
        )

        assert math.isclose(mag_point.mlat.deg, 20, rel_tol=1e-3)  # deg
        assert math.isclose(mag_point.mlat.value, 0.349065852, rel_tol=1e-8)  # rad
        assert mag_point.l_shell == 5.0
        assert mag_point.planetary_radius.value == 6000000.0  # meters
        assert mag_point.mag_dipole_moment.value == 7e15  # T*m^3
        assert math.isclose(mag_point.flux_density.value, 4.37659478e-07)  # T

    def test_magpoint_5(self):
        """Raise TypeError. mlat has no units"""
        mlat_deg = 0.349065852
        l_shell = 5.0 * u.dimensionless_unscaled

        with pytest.raises(TypeError):
            MagPoint(mlat_deg, l_shell)

    def test_magpoint_6(self):
        """Raise u.UnitsError. mlat has wrong units"""
        mlat_deg = 0.349065852 * u.m  # not convertible to radians
        l_shell = 5.0 * u.dimensionless_unscaled

        with pytest.raises(u.UnitsError):
            MagPoint(mlat_deg, l_shell)
