import numpy as np
from astropy import units as u
from astropy.units import Quantity

from piran.helpers import get_real_and_positive_roots
from piran.magpoint import MagPoint


class Bounce:
    """
    This class provides methods and attributes necessary for calculating
    diffusion coefficients averaged over particle bounce periods, following the
    methodology outlined in Glauert & Horne 2005 (section 4).

    Parameters
    ----------
    equatorial_pitch_angle: Quantity[u.rad]
        Equatorial pitch angle

    equatorial_magpoint: MagPoint
        Equatorial magnetic point object
    """

    @u.quantity_input
    def __init__(
        self,
        equatorial_pitch_angle: Quantity[u.rad],
        equatorial_magpoint: MagPoint,
    ) -> None:
        self.__equatorial_pitch_angle = equatorial_pitch_angle.to(u.rad)
        self.__equatorial_magpoint = equatorial_magpoint

        if not 0.0 < self.__equatorial_pitch_angle.value < (np.pi / 2):
            msg = "Equatorial pitch angle must be between 0 and 90 degrees, exclusive."
            raise ValueError(msg)

        if self.__equatorial_magpoint.mlat != (0.0 << u.rad):
            msg = "The equatorial magpoint must have magnetic latitude of 0 radians."
            raise ValueError(msg)

        self.__particle_bounce_period = self.get_particle_bounce_period()
        self.__mirror_latitude = self.get_mirror_latitude()

    @property
    def particle_bounce_period(self):
        return self.__particle_bounce_period

    @property
    def mirror_latitude(self):
        return self.__mirror_latitude

    @u.quantity_input
    def get_particle_bounce_period(self) -> Quantity[u.dimensionless_unscaled]:
        r"""
        Approximate the particle bounce period :math:`T(\alpha_{eq})`
        as described in Glauert & Horne 2005 (equation 27).

        Returns
        -------
        Quantity[u.dimensionless_unscaled]
            Particle bounce period
        """
        return 1.3 - 0.56 * np.sin(self.__equatorial_pitch_angle)

    @u.quantity_input
    def get_mirror_latitude(self) -> Quantity[u.rad]:
        r"""
        Calculate the mirror latirude :math:`\lambda_m`
        as described in Glauert & Horne 2005 (equation 28).

        Returns
        -------
        Quantity[u.rad]
            Mirror latitude
        """
        sine4_a_eq = np.sin(self.__equatorial_pitch_angle) ** 4
        p = np.polynomial.Polynomial([-4 * sine4_a_eq, 3 * sine4_a_eq, 0, 0, 0, 0, 1])
        roots = p.roots()

        real_pos_roots = get_real_and_positive_roots(roots)

        if real_pos_roots.size == 0:
            mirror_latitude = np.nan
        elif real_pos_roots.size == 1:
            mirror_latitude = np.arccos(np.sqrt(real_pos_roots[0]))
        else:
            msg = "The mirror latitude equation returned multiple valid roots."
            raise AssertionError(msg)

        return mirror_latitude << u.rad

    @u.quantity_input
    def get_bounce_pitch_angle(
        self,
        mlat: Quantity[u.rad],
    ) -> Quantity[u.rad]:
        """
        Given equatorial pitch angle calculates the pitch angle for a given
        magnetic latitude.

        This method leverages the conservation of magnetic moment (equation 4.13 in
        "Chapter 4. Adiabatic Invariants, Introduction to Geomagnetically Trapped Radiation").
        The pitch angle is determined using Equation 4.22.

        **Note:** When the given magnetic latitude is the mirror latitude, the
        quantity inside the arcsin function should ideally be 1. However, due
        to floating-point precision limitations, this value might slightly
        exceed 1, leading to NaN results. To address this, we could consider
        returning π/2 to account for the maximum pitch angle, but our current
        implementation does not handle this scenario.

        Parameters
        ----------
        mlat : Quantity[u.rad]
            The magnetic latitude, given in units convertible to radians.

        Returns
        -------
        Quantity[u.rad]
            Pitch angle for given magnetic latitude
        """
        new_magpoint = MagPoint(
            mlat.to(u.rad),
            self.__equatorial_magpoint.l_shell,
            self.__equatorial_magpoint.planetary_radius,
            self.__equatorial_magpoint.mag_dipole_moment,
        )
        pitch_angle = np.arcsin(
            np.sin(self.__equatorial_pitch_angle)
            * np.sqrt(
                new_magpoint.flux_density / self.__equatorial_magpoint.flux_density
            )
        )

        return pitch_angle
