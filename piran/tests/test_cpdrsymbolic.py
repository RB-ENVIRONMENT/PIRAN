import math

from piran.cpdrsymbolic import CpdrSymbolic


class TestCpdrSymbolic:
    def test_cpdrsymbolic_1(self):
        # Neutral electron and proton plasma
        # mlat = 0 degrees
        # l_shell = 4.5
        # fpe/fce = 1.5
        omega_ce = -59760.22111062
        omega_cp = 32.54643362
        omega_pe = 89640.33166593
        omega_pp = 2091.93920967

        # E = 1MeV
        # alpha = 5 degrees
        v_par = 2.81054871e08
        gamma = 2.95695118

        # Values that solve both the resonant condition
        # and the dispersion relation.
        n = 2
        X = 1.0
        psi = math.atan(X)
        omega = 19801.87442897178
        k = 0.0003030255929269047

        cpdr_sym = CpdrSymbolic(2)

        values_dict = {
            "v_par": v_par,
            "gamma": gamma,
            "omega_c": (omega_ce, omega_cp),
            "omega_p": (omega_pe, omega_pp),
            "n": n,
            "X": X,
            "psi": psi,
            "omega": omega,
            "k": k,
        }

        poly_in_k_eval = cpdr_sym.poly_in_k.subs(values_dict)
        resonant_poly_in_omega_eval = cpdr_sym.resonant_poly_in_omega.subs(values_dict)

        assert cpdr_sym.n_species == 2
        assert math.isclose(poly_in_k_eval, 0.0, abs_tol=1e-10)
        assert math.isclose(resonant_poly_in_omega_eval, 0.0, abs_tol=1e-10)

    def test_cpdrsymbolic_2(self):
        # Neutral electron and proton plasma
        # mlat = 0 degrees
        # l_shell = 4.5
        # fpe/fce = 1.5
        omega_ce = -59760.22111062
        omega_cp = 32.54643362
        omega_pe = 89640.33166593
        omega_pp = 2091.93920967

        # E = 1MeV
        # alpha = 5 degrees
        v_par = 2.81054871e08
        gamma = 2.95695118

        # Values that solve both the resonant condition
        # and the dispersion relation.
        n = -2
        X = 1.0
        psi = math.atan(X)
        omega = 8859.49231799109
        k = 0.00015880715819148176

        cpdr_sym = CpdrSymbolic(2)

        values_dict = {
            "v_par": v_par,
            "gamma": gamma,
            "omega_c": (omega_ce, omega_cp),
            "omega_p": (omega_pe, omega_pp),
            "n": n,
            "X": X,
            "psi": psi,
            "omega": omega,
            "k": k,
        }

        poly_in_k_eval = cpdr_sym.poly_in_k.subs(values_dict)
        resonant_poly_in_omega_eval = cpdr_sym.resonant_poly_in_omega.subs(values_dict)

        assert cpdr_sym.n_species == 2
        assert math.isclose(poly_in_k_eval, 0.0, abs_tol=1e-8)
        assert math.isclose(resonant_poly_in_omega_eval, 0.0, abs_tol=1e-8)
