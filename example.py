from sympy import (
    symbols,
    Sum,
    IndexedBase,
    Idx,
    cos,
    product,
    init_printing,
    collect,
    expand,
    pprint,
    Pow,
    simplify,
)

init_printing(pretty_print=True)

# Use this for indexing
i = symbols("i", cls=Idx)

# PARTICLE_SPECIES defines total number of particle species in plasma
# e.g. for proton-electron plasma, PARTICLE_SPECIES = 2
PARTICLE_SPECIES = 2
#
# These are used for convenience
PS_INDEX = PARTICLE_SPECIES - 1
PS_RANGE = (i, 0, PS_INDEX)

# Define lots of algebraic symbols
A, B, C, X, R, L, P, S, omega = symbols("A, B, C, X, R, L, P, S, omega")

# Indexed symbols (one per particle species)
Omega_Base = IndexedBase("Omega_Base")
Omega_i = Omega_Base[i]

omega_p_base = IndexedBase("omega_p")
omega_p_i = omega_p_base[i]

# Substitution of the resonance condition into the CPDR yields an expression
# in negative powers of omega.
# Multiply by this thing to remove omega from the denominator of all terms
# in our expression.
MULTIPLICATION_FACTOR = Pow(omega, 6) * product(
    (omega + Omega_i) * (omega - Omega_i), PS_RANGE
)

# Stix Parameters
# Use .doit() to force expansion of the sum, so that multiplication by
# MULTIPLICATION_FACTOR (to be completed shortly) properly removes all
# traces of omega from the denominator of each term
R = 1 - Sum((omega_p_i**2) / (omega * (omega + Omega_i)), PS_RANGE).doit()
L = 1 - Sum((omega_p_i**2) / (omega * (omega - Omega_i)), PS_RANGE).doit()
P = 1 - Sum((omega_p_i**2) / (omega**2), PS_RANGE).doit()
S = (R + L) / 2

# CPDR = A*mu**4 - B*mu**2 + C
# Use MULTIPLICATION_FACTOR pre-emptively here with 'simplify' to force SymPy
# to remove omega from the denominator of each term.
# NB. 'simplify' is a very non-targeted way of doing this; it 'works', but I'd
# be much more comfortable if we were using something more specific!
A = simplify(MULTIPLICATION_FACTOR * (S * (X**2) + P))
B = simplify(MULTIPLICATION_FACTOR * (R * L * (X**2) + P * S * (2 + (X**2))))
C = simplify(MULTIPLICATION_FACTOR * (P * R * L * (1 + (X**2))))

# Print A, B, C to check that they are in the form we expect
pprint(f"A = {A}")
print("\n\n")
pprint(f"B = {B}")
print("\n\n")
pprint(f"C = {C}")
print("\n\n")

# More symbols for mu
# NB. mu has *another* instance of omega in the denominator, so we're going to
# need to ask SymPy to simplify our expression again...
c, v_par, psi, n, gamma, mu = symbols("c, v_par, psi, n, gamma, mu")
mu = (c / (v_par * cos(psi))) * (1 - (n * Omega_Base[0] / (gamma * omega)))

# Compose the CPDR from its constituent parts, calling 'simplify' on each
# individually to avoid overloading SymPy.
CPDR = symbols("CPDR")

CPDR_1 = simplify(A * Pow(mu, 4))
pprint(f"CPDR_1 = {CPDR_1}")
print("\n\n")

CPDR_2 = simplify(-B * Pow(mu, 2))
pprint(f"CPDR_2 = {CPDR_2}")
print("\n\n")

CPDR_3 = simplify(C)
pprint(f"CPDR_3 = {CPDR_3}")
print("\n\n")

# Pull everything together, request polynomial form, and print!
CPDR = collect(expand(CPDR_1 + CPDR_2 + CPDR_3), omega).as_poly(omega)
pprint(f"CPDR = {CPDR}")
print("\n\n")

pprint(CPDR.as_dict())
print("\n\n")

# We now have a polynomial representation for the CPDR in terms of omega.
# Unfortunately, neither of the following work...
#
# pprint(solve(CPDR, omega))
# pprint(roots(CPDR, omega))
