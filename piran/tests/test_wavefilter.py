import pytest
from astropy import units as u
from astropy.coordinates import Angle

from piran.magpoint import MagPoint
from piran.plasmapoint import PlasmaPoint
from piran.stix import Stix
from piran.wavefilter import WaveFilter


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
