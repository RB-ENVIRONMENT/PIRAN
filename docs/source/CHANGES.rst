Version 1.1.1
=============

Packaging / Dependency Changes
------------------------------
-  Specified minimum and maximum version bounds for all dependencies and
   restricted `requires-python` to `>=3.10,<3.13` so that piran works as
   expected with the current ecosystem:

   -  `plasmapy>=2024.10.0,<2027`
   -  `astropy>=6.1,<9`
   -  `numpy>=1.26,<2.6`
   -  `scipy>=1.11,<1.18`
   -  `matplotlib>=3.8,<3.12`

   The critical upper bound is `scipy<1.18`: starting from scipy 1.18.0 the
   `scipy.integrate.simpson` routine no longer strips units from astropy
   `Quantity` inputs (previously it returned a plain float, while `trapezoid`
   always returned a `Quantity`; both now preserve units). This broke the
   Glauert normalisation factor computation in `normalisation.py` and the
   bounce-averaged integrals over magnetic latitude.
-  Removed `sympy` from the dependencies since it is no longer used anywhere
   in the codebase.
-  Bounded the development/test tool dependencies (`ruff<0.16`, `black<27`,
   `isort<10`, `pytest<10`, `coverage<8`): ruff 0.16.0 introduced new lint
   rules that flag pre-existing code. The code will be cleaned up and these
   bounds raised in a future PR.

Tests
-----
-  Loosened (only where necessary) the `math.isclose` tolerances of a few unit
   tests. astropy 8.0 adopted the CODATA-2022 physical constants, which shifts
   e.g. the electron mass by ~1.4e-9 relative; this made 12 tests, whose
   `math.isclose` relative tolerances were tighter than this drift (the
   default `rel_tol=1e-9` in most cases), fail with newer astropy, without any
   change in the underlying physics.

Version 1.1.0
=============

API Changes
-----------
-  The `freq_cutoff_params` parameter for the `Cpdr` class has been changed from a `list`
   to a `Gaussian` object. A convenience method, `gauss.from_gyrofrequency_params`, has
   been added to help preserve the old functionality. This change has been motivated by the
   need to fix the wave frequency spectrum parameters (with respect to latitude) when computing
   bounce-averaged diffusion coefficients in bounce_averaged_diffusion_coefficients.py.


Bug Fixes
---------
-  For bounce averaged calculations the wave frequency spectrum parameters are now fixed with 
   respect to latitude (and are defined in terms of equatorial gyrofrequency in the input file).
-  The lower bound for the `WhistlerFilter` class in wavefilter.py is now defined as 
   the maximum of the lower hyrbid frequency and the proton gyrofrequency. In addition there is 
   a new constraint that the wave resonance cone angle is a real number.
-  For bounce averaged calculations the number density is now fixed with respect to latitude in 
   bounce_averaged_diffusion_coefficients.py (and is defined via the ratio of the equatorial 
   plasma frequency to the equatorial gyrofrequency in the input file). 

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

- Added the `mlat_cutoff` parameter to the input specification, allowing users to set a maximum
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

- Added a warning message that is printed when the resonance cone angle is imaginary.

Version 1.0.0
=============

- Initial release.
