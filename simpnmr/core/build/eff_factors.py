# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute effective electronic g-factors and angular momenta.

Provides helpers to evaluate effective g-values and angular momentum parameters
from spin, orbital, and total angular momentum quantum numbers.
Also provides rank-2 Stevens operator-equivalent factors (α_J) for
lanthanide/actinide J-multiplet states.
"""

from fractions import Fraction

from simpnmr.core.const.physics import GE


# ---------------------------------------------------------------------------
# Hund's-rule Stevens factor helpers
# ---------------------------------------------------------------------------

def _hund_occupation(l: int, n: int) -> list[int]:
    """Return the list of m_l values for the Hund's-rule ground state of l^n.

    Fills the first half of the shell from m_l = l downward (spin-up), then
    begins filling the second half from m_l = l downward (spin-down).

    Args:
        l: Orbital angular momentum quantum number (e.g. 3 for f-shell).
        n: Number of electrons (1 ≤ n ≤ 2*(2l+1)).

    Returns:
        List of m_l values, one entry per electron.
    """
    half = 2 * l + 1
    if n <= half:
        return list(range(l, l - n, -1))
    else:
        k = n - half
        base = list(range(-l, l + 1))   # all m_l singly filled → net M_L = 0
        extra = list(range(l, l - k, -1))  # k spin-down electrons from top
        return base + extra


def _infer_fshell_n(L: int, S, J, l: int = 3) -> int:
    """Map a Hund's-rule (L, S, J) ground term to the electron count n.

    Uses the fact that S = n/2 for the first half of the shell and
    S = (2*(2l+1) - n)/2 for the second half.  The Hund's-rule coupling
    determines which half the configuration belongs to:

    * J = L + S  → second half (J maximised)
    * J = |L - S| → first half (J minimised)

    Args:
        L: Total orbital angular momentum.
        S: Total spin quantum number (int or Fraction).
        J: Total angular momentum quantum number (int or Fraction).
        l: Shell angular momentum (default 3 for f-shell).

    Returns:
        Electron count n.

    Raises:
        ValueError: If J is neither L+S nor |L−S|.
    """
    S = float(Fraction(S).limit_denominator(10 ** 6))
    J = float(Fraction(J).limit_denominator(10 ** 6))
    n_first = int(round(2 * S))
    max_n = 2 * (2 * l + 1)
    if abs(J - (L + S)) < 1e-9:        # second half of shell
        return max_n - n_first
    if abs(J - abs(L - S)) < 1e-9:     # first half of shell
        return n_first
    raise ValueError(
        f"Cannot infer electron count from L={L}, S={S}, J={J}: "
        "J is neither L+S nor |L−S|."
    )


def alpha_L_fshell(n: int, l: int = 3) -> Fraction:
    """Orbital Stevens rank-2 factor α_L for the l^n Hund's ground LS-term.

    Computes the intrashell expectation value of the rank-2 Stevens operator
    O₂⁰ normalised to ⟨r²⟩, evaluated in the maximum-ML state of the
    Hund's-rule ground LS-term.

    Args:
        n: Number of electrons.
        l: Shell angular momentum (default 3 for f-shell).

    Returns:
        α_L as an exact Fraction.
    """
    occ = _hund_occupation(l, n)
    L = sum(occ)
    if L == 0:
        return Fraction(0)
    sum_m2 = sum(m * m for m in occ)
    num = 2 * (l * (l + 1) * n - 3 * sum_m2)
    den = (2 * l - 1) * (2 * l + 3)
    return Fraction(num, den * L * (2 * L - 1))


def _clebsch_gordan(j1, m1, j2, m2, j, m) -> float:
    """Clebsch-Gordan coefficient ⟨j₁ m₁; j₂ m₂ | j m⟩.

    Uses sympy for exact evaluation; returns a Python float.

    Args:
        j1, m1: First angular momentum and its projection.
        j2, m2: Second angular momentum and its projection.
        j, m:   Coupled angular momentum and its projection.

    Returns:
        Real-valued CG coefficient.
    """
    from sympy import Rational
    from sympy.physics.quantum.cg import CG

    def _r(x):
        return Rational(Fraction(x).limit_denominator(10 ** 6))

    return float(CG(_r(j1), _r(m1), _r(j2), _r(m2), _r(j), _r(m)).doit())


def alpha_J_from_LSJ(L: int, S, J, l: int = 3) -> float:
    """Rank-2 Stevens factor α_J from (L, S, J) for the Hund's ground multiplet.

    Evaluates the reduced matrix element of the rank-2 Stevens operator
    O₂⁰ in the |J, M_J=J⟩ state of the l^n Hund's-rule ground multiplet,
    assuming the ion is in the l-shell (default l = 3, f-shell).

    The relation used is::

        α_J · J(2J−1) = α_L · Σ_{M_L} |⟨L M_L; S M_S | J J⟩|² (3 M_L² − L(L+1))

    where M_S = J − M_L and α_L = alpha_L_fshell(n, l).

    Args:
        L: Total orbital angular momentum quantum number.
        S: Total spin quantum number S (int, float, or Fraction).
        J: Total angular momentum quantum number J (int, float, or Fraction).
        l: Shell angular momentum (default 3 for f-shell).

    Returns:
        α_J as a float (0.0 if L = 0).
    """
    S = Fraction(S).limit_denominator(10 ** 6)
    J = Fraction(J).limit_denominator(10 ** 6)

    if L == 0:
        return 0.0

    n = _infer_fshell_n(L, S, J, l)
    aL = alpha_L_fshell(n, l)

    bracket = 0.0
    M_L = -L
    while M_L <= L:
        M_S = J - M_L
        if abs(M_S) <= S and (S - M_S).denominator == 1:
            c = _clebsch_gordan(L, M_L, S, M_S, J, J)
            bracket += c * c * (3 * M_L * M_L - L * (L + 1))
        M_L += 1

    JJ = float(J) * (2 * float(J) - 1)
    return float(aL) * bracket / JJ


def calc_g_eff(spin: float, orbit: float, total_momentum_J: float | None):
    """Computes an effective electron g-factor.

    For spin-only systems (transition metals, organic radicals) where no total J is
    defined, this returns the free-electron g value `GE`.

    For systems with well-defined ``L``, ``S``, and ``J`` (e.g. lanthanides), this
    returns the Landé ``g_J`` factor.

    Args:
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum quantum number ``J``. If ``None`` or
            ``0``, the function falls back to `GE`.

    Returns:
        Effective g-factor (either `GE` or ``g_J``).
    """

    # Spin-only case: no total J provided or explicitly zero
    if total_momentum_J is None or total_momentum_J == 0.0:
        return GE

    # Landé g_J expression using S, L and J
    J = float(total_momentum_J)

    return 1.5 + (spin * (spin + 1) - orbit * (orbit + 1)) / (2.0 * J * (J + 1))


def choose_S_eff(spin: float, total_momentum_J: float | None):
    """Returns the effective angular momentum quantum number used in prefactors.

    Args:
        spin: Spin quantum number ``S``.
        total_momentum_J: Total angular momentum ``J``. If ``None``, `spin` is used.

    Returns:
        ``S`` for spin-only systems, or ``J`` when `total_momentum_J` is provided.
    """

    return spin if total_momentum_J is None else total_momentum_J
