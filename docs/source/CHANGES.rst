Version 1.0.1
=============

New Features
------------

piran/plasmapoint.py
^^^^^^^^^^^^^^^^^^^^

- Added a `lower_hybrid_freq` property to enable users to conveniently obtain the lower
  hybrid frequency for each particle in a plasma without the need to (re-)calculate this
  separately.

src/scripts/bounce_averaged_diffusion_coefficients.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Added the `mlat_cutoff` parameter to the input specification, allowins users to set a maximum
  magnetic latitude beyond which it is assumed that no waves are present.

Bug Fixes
---------

piran/wavefilter.py
^^^^^^^^^^^^^^^^^^^

- Used the newly added `PlasmaPoint.lower_hybrid_freq` property to set a lower bound
  (the lower hybrid frequency for protons) on the acceptable frequency range in `WhistlerFilter`.
  There is an implicit assumption in here that we are only looking at electron-proton plasmas,
  which will need to be addressed when we improve the support for more complicated plasma
  compositions in the future.

src/scripts/bounce_averaged_diffusion_coefficients.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Cap the upper limit on the range of integration over the magnetic latitude to 0.9999 times
  the mirror point, to ensure we use a consistent approach for the integrable singularity at the
  mirror point. Previously, when calculating the integrand at the mirror point
  `Bounce.get_bounce_pitch_angle` would return either exactly `pi/2` or a value very close to it
  (approaching from below), dependent on floating point precision. The `pi/2` pitch angle gets
  skipped (for being singular) whereas anything else is included. Capping the magnetic latitude at
  0.9999 times the mirror point ensures that we don't have to deal with this ambiguity. A more
  robust long-term solution might be to re-examine the behaviour of `Bounce.get_bounce_pitch_angle`
  and look for a way to more accurately evaluate the integrand at the mirror point.

Other Changes and Additions
---------------------------

src/scripts/bounce_averaged_diffusion_coefficients.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Added a warning message that is printed when the resonance cone angle is negative.

Version 1.0.0
=============

- Initial release.
