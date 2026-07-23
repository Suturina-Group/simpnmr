.. _theory:

Theory
======

Below, we outline the key models and equations implemented in ``simpnmr``.

.. _TOTAL:

Total shift
-----------

In ``simpnmr``, the diamagnetic, spin-dipolar, Fermi-contact,
Fermi-contact g-correction, and orbital contributions are treated as separate
components of the total shift. The total
predicted chemical shift is therefore written as

.. math::
   :label: :eq: total_orb

    \delta^{\mathrm{TOTAL}}=\delta^{\mathrm{DIA}}+\delta^{\mathrm{SD}}+\delta^{\mathrm{FC}}+\delta^{\mathrm{FC}}_{\mathrm{g-corr}}+\delta^{\mathrm{ORB}}_{\mathrm{iso}}+\delta^{\mathrm{ORB}}_{\mathrm{aniso}}

.. note::

    The calculation without :math:`\delta^{\mathrm{ORB}}` remains physically reasonable
    because the orbital contribution is expected to become negligible at sufficiently
    large distances from the paramagnetic centre. [Lang2020]_


.. _DIA:

Diamagnetic shift
-----------------

When the diamagnetic contribution is obtained from DFT shielding data, a
reference value is required in order to convert shielding into a chemical
shift. In ``simpnmr``, the diamagnetic shift contribution is written as

.. math::
   :label: :eq: dia

    \delta^{\mathrm{DIA}}=\sigma_{\mathrm{ref}}-\sigma

where :math:`\sigma` is the calculated shielding for the nucleus of interest
and :math:`\sigma_{\mathrm{ref}}` is the corresponding reference shielding.

This diamagnetic term is then added directly to the other shift contributions
when forming :math:`\delta^{\mathrm{TOTAL}}`.

.. note::

   For a consistent diamagnetic shift, the calculated shielding
   :math:`\sigma` and the reference shielding :math:`\sigma_{\mathrm{ref}}`
   must be obtained at the same level of theory.

.. _PCS:

Spin-dipolar contribution
-------------------------

In ``simpnmr``, the contribution arising from the anisotropic magnetic
susceptibility and the traceless spin-dipolar hyperfine interaction is treated
as a separate spin-dipolar shift channel, denoted
:math:`\delta^{\mathrm{SD}}`. This term depends on the anisotropic part of
the magnetic susceptibility tensor and the spin-dipolar part of the hyperfine
coupling tensor (HFC).

1. Spin-dipolar contribution with hyperfine from DFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To account for the non-point nature of the paramagnetic centre, the normalised
traceless spin-dipolar hyperfine contribution can be obtained from a simple
single-point DFT calculation of :math:`\mathbf{A}^{\mathrm{SD}}`:

.. math::
   :label: :eq: pcs

   \delta^{\mathrm{SD}}=\frac{1}{3} \operatorname{tr}\left(\Delta \boldsymbol{\chi} \cdot \mathbf{A}^{\mathrm{SD}}\right)

In ``simpnmr``, this defines the spin-dipolar contribution independently of
the FC, FC g-correction, and orbital shift terms.

.. note::

   The spin-dipolar hyperfine tensor :math:`\mathbf{A}^{\mathrm{SD}}` is
   always traceless.

2. Spin-dipolar contribution with point-dipole approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming that a paramagnetic metal centre is at the origin and a nucleus of
interest has coordinates :math:`(x, y, z)`, the spin-dipolar contribution
:math:`\delta^{\mathrm{SD}}` can be calculated as a third of the trace of
the magnetic susceptibility tensor :math:`\chi` multiplied by the traceless
spin-dipolar hyperfine tensor, which in the point-dipole approximation is a
matrix that depends only on the nuclear coordinates:

.. math::
   :label: :eq: pcs_pd

    \delta^{\mathrm{SD}}=\frac{1}{12 \pi r^5} \operatorname{tr}\left[\left(\begin{array}{ccc}
    \chi_{x x} & \chi_{x y} & \chi_{x z} \\
    \chi_{y x} & \chi_{y y} & \chi_{y z} \\
    \chi_{z x} & \chi_{z y} & \chi_{z z}
    \end{array}\right) \cdot\left(\begin{array}{ccc}
    3 x^2-r^2 & 3 x y & 3 x z \\
    3 x y & 3 y^2-r^2 & 3 y z \\
    3 x z & 3 y z & 3 z^2-r^2
    \end{array}\right)\right]

If the coordinates are specified in Å and :math:`\chi` is in Å\ :sup:`3`,
then the equation above, multiplied by 10\ :sup:`6`, gives the spin-dipolar
contribution in ppm.

.. _FC:

Fermi-contact shift
-------------------

The Fermi-contact contribution is split into a spin-only term
:math:`\delta^{\mathrm{FC}}` and an additional g-correction term
:math:`\delta^{\mathrm{FC}}_{\mathrm{g-corr}}`. Both depend on the isotropic
Fermi-contact hyperfine interaction at the nucleus.

.. note::

   By construction, the Fermi-contact hyperfine tensor
   :math:`\mathbf{A}^{\mathrm{FC}}` is isotropic. In matrix form, it is
   diagonal with equal diagonal elements.

1. FC with spin-only magnetic susceptibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the simplest model, the Fermi-contact shift is proportional to the
isotropic Fermi-contact hyperfine interaction at the nucleus of interest and
the spin-only magnetic susceptibility:

.. math::
   :label: :eq: fc_s

    \delta^{\mathrm{FC}}=\chi_{iso}^S A^{FC}

In ``simpnmr``, :math:`\delta^{\mathrm{FC}}` is evaluated from the isotropic
part of the Fermi-contact hyperfine tensor, i.e. from
:math:`\frac{1}{3}\operatorname{tr}(\mathbf{A}^{\mathrm{FC}})`, where the spin-only magnetic susceptibility in SI units is:

.. math::
   :label: :eq: chi_s

    \chi_{iso}^S=\frac{\mu_0 \mu_B^2 \mathrm{g}_{\mathrm{e}}^2 S(S+1)}{3 k T}

where :math:`\mu_0` is the vacuum permeability, :math:`\mu_B` is the Bohr
magneton, :math:`\mathrm{g}_{\mathrm{e}}` is the free-electron g-factor,
:math:`S` is the total spin, :math:`k` is the Boltzmann constant, and
:math:`T` is the temperature.

2. FC g-correction from g-corrected magnetic susceptibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to account for the effect of :math:`\mathbf{g}_{\mathrm{ab-initio}}`
anisotropy on the FC term, ``simpnmr`` treats the corresponding correction as
a separate contribution.

.. math::
    :label: :eq: FC_g

    \delta^{\mathrm{FC}}_{\mathrm{g-corr}}=\left[\chi^{\mathrm{g-corr}}_{\mathrm{iso}}-\chi^S_{\mathrm{iso}}\right]\,\frac{1}{3}\operatorname{tr}\left(\mathbf{A}^{\mathrm{FC}}\right)

where the g-corrected isotropic susceptibility is

.. math::
   :label: :eq: chi_g_corr

    \chi^{\mathrm{g-corr}}_{\mathrm{iso}}=\frac{g_{\mathrm{e}}}{3}\left(\frac{\chi_x}{g_x}+\frac{\chi_y}{g_y}+\frac{\chi_z}{g_z}\right)

This correction remains proportional to the spin-only Fermi-contact hyperfine
term and isolates the additional contribution arising from g-tensor
anisotropy.

.. note::

   Here, :math:`\mathbf{g}_{\mathrm{ab-initio}}` should be taken from the same
   level of theory as the susceptibility tensor used to compute
   :math:`\chi^{\mathrm{g-corr}}_{\mathrm{iso}}`.

.. _ORB:

Orbital shift contribution
--------------------------

In ``simpnmr``, the orbital contribution is treated as two additional shift
channels, :math:`\delta^{\mathrm{ORB}}_{\mathrm{iso}}` and
:math:`\delta^{\mathrm{ORB}}_{\mathrm{aniso}}`. These do not modify the
definitions of :math:`\delta^{\mathrm{SD}}`, :math:`\delta^{\mathrm{FC}}`, or
:math:`\delta^{\mathrm{FC}}_{\mathrm{g-corr}}`.

.. note::

   The orbital contribution is evaluated only when both of the following are
   available from the same QC source:

   - the orbital hyperfine contribution :math:`\mathbf{A}^{\mathrm{ORB}}`
   - the associated :math:`\mathbf{g}_{\mathrm{DFT}}` tensor

If the orbital hyperfine contribution :math:`\mathbf{A}^{\mathrm{ORB}}` and the
associated :math:`\mathbf{g}_{\mathrm{DFT}}` tensor are available from the
same QC source, the isotropic orbital contribution is evaluated as

.. math::
   :label: :eq: orb_iso

    \delta^{\mathrm{ORB}}_{\mathrm{iso}}=
    \chi_{\mathrm{iso}}\frac{1}{3}\operatorname{tr}\left[\frac{g_{\mathrm{e}}}{\mathbf{g}^{\mathrm{T}}_{\mathrm{DFT}}}\left(\mathbf{A}^{\mathrm{SD}}+\mathbf{A}^{\mathrm{ORB}}\right)^{\mathrm{T}}\right]

and the anisotropic orbital contribution is evaluated as

.. math::
   :label: :eq: orb_aniso

    \delta^{\mathrm{ORB}}_{\mathrm{aniso}}=\frac{1}{3}\operatorname{tr}\left[\Delta\boldsymbol{\chi}\frac{g_{\mathrm{e}}}{\mathbf{g}^{\mathrm{T}}_{\mathrm{DFT}}}\left(\mathbf{A}^{\mathrm{SD}}+\mathbf{A}^{\mathrm{ORB}}\right)^{\mathrm{T}}-\Delta\boldsymbol{\chi}\mathbf{A}^{\mathrm{SD}}\right]

For reporting purposes, the total orbital shift contribution is the sum

.. math::
   :label: :eq: orb_total

    \delta^{\mathrm{ORB}}=\delta^{\mathrm{ORB}}_{\mathrm{iso}}+\delta^{\mathrm{ORB}}_{\mathrm{aniso}}

Therefore, evaluating orbital shift contributions requires both the orbital
hyperfine contribution and the associated :math:`\mathbf{g}_{\mathrm{DFT}}`
tensor from the same QC source.

.. note::

   Another common source of confusion is the role of
   :math:`\mathbf{A}^{\mathrm{SD}}` in the orbital expressions above. In
   ``simpnmr``, :math:`\mathbf{A}^{\mathrm{SD}}` still defines the
   spin-dipolar term on its own, while the orbital term is evaluated separately
   from the transformed combination
   :math:`\mathbf{A}^{\mathrm{SD}}+\mathbf{A}^{\mathrm{ORB}}`.

.. _RELAX:

Paramagnetic relaxation
-----------------------

``simpnmr`` evaluates nucleus-resolved longitudinal (:math:`R_1`) and
transverse (:math:`R_2`) paramagnetic relaxation rates from the
Solomon–Bloembergen–Morgan (SBM) dipolar and Fermi-contact terms and the
Guéron Curie-spin term. All terms share the Lorentzian spectral density

.. math::

    J(\omega,\tau) = \frac{\tau}{1+\omega^2\tau^2}

where :math:`\omega_I` and :math:`\omega_S` are the nuclear and electron
Larmor angular frequencies, :math:`r` is the electron–nucleus distance,
:math:`\gamma_I` is the nuclear gyromagnetic ratio, :math:`A_{\mathrm{iso}}`
is the isotropic Fermi-contact coupling (in angular-frequency units), and
:math:`T` is the temperature. The correlation times are the dipolar
correlation times :math:`\tau_{c1}, \tau_{c2}`, the electronic correlation
times :math:`\tau_{e1}, \tau_{e2}`, and the rotational correlation time
:math:`\tau_R`.

.. note::

   :math:`g_{\mathrm{eff}}` and :math:`S_{\mathrm{eff}}` are the effective
   electron *g*-factor and angular-momentum quantum number. For spin-only
   systems they are the free-electron value :math:`g_e` and the spin
   :math:`S`; for systems with a well-defined total angular momentum they are
   the Landé :math:`g_J` and :math:`J`.

SBM dipolar relaxation
^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    R_1^{\mathrm{DD}} = \frac{2}{15}\left(\frac{\mu_0}{4\pi}\right)^2
    \frac{\gamma_I^2\, g_{\mathrm{eff}}^2\, \mu_B^2\,
    S_{\mathrm{eff}}(S_{\mathrm{eff}}+1)}{r^6}
    \Big[\,3\,J(\omega_I,\tau_{c1}) + 6\,J(\omega_I+\omega_S,\tau_{c2})
    + J(\omega_I-\omega_S,\tau_{c2})\,\Big]

.. math::

    R_2^{\mathrm{DD}} = \frac{1}{15}\left(\frac{\mu_0}{4\pi}\right)^2
    \frac{\gamma_I^2\, g_{\mathrm{eff}}^2\, \mu_B^2\,
    S_{\mathrm{eff}}(S_{\mathrm{eff}}+1)}{r^6}
    \Big[\,4\,J(0,\tau_{c1}) + 3\,J(\omega_I,\tau_{c1})
    + 6\,J(\omega_S,\tau_{c2}) + 6\,J(\omega_I+\omega_S,\tau_{c2})
    + J(\omega_I-\omega_S,\tau_{c2})\,\Big]

SBM Fermi-contact relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    R_1^{\mathrm{SC}} = \frac{2}{3}\, A_{\mathrm{iso}}^2\,
    S_{\mathrm{eff}}(S_{\mathrm{eff}}+1)\, J(\omega_I-\omega_S,\tau_{e2})

.. math::

    R_2^{\mathrm{SC}} = \frac{1}{3}\, A_{\mathrm{iso}}^2\,
    S_{\mathrm{eff}}(S_{\mathrm{eff}}+1)
    \Big[\,J(0,\tau_{e1}) + J(\omega_I-\omega_S,\tau_{e2})\,\Big]

Guéron Curie relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^

The Curie-spin term uses the point-dipole approximation and the
thermally averaged (static) electron moment:

.. math::

    R_1^{\mathrm{Curie}} = \frac{2}{5}\left(\frac{\mu_0}{4\pi}\right)^2
    \frac{\omega_I^2\, g_{\mathrm{eff}}^4\, \mu_B^4\,
    \big[S_{\mathrm{eff}}(S_{\mathrm{eff}}+1)\big]^2}{(3 k_B T)^2\, r^6}\,
    3\,J(\omega_I,\tau_R)

.. math::

    R_2^{\mathrm{Curie}} = \frac{1}{5}\left(\frac{\mu_0}{4\pi}\right)^2
    \frac{\omega_I^2\, g_{\mathrm{eff}}^4\, \mu_B^4\,
    \big[S_{\mathrm{eff}}(S_{\mathrm{eff}}+1)\big]^2}{(3 k_B T)^2\, r^6}
    \Big[\,4\,J(0,\tau_R) + 3\,J(\omega_I,\tau_R)\,\Big]

.. note::

   For both the SBM dipolar and the Curie terms the :math:`R_1` prefactor is
   twice the :math:`R_2` prefactor (:math:`\tfrac{2}{15}` vs
   :math:`\tfrac{1}{15}`, and :math:`\tfrac{2}{5}` vs :math:`\tfrac{1}{5}`).
   In the high-field, fast-motion limit (:math:`\omega_S\tau \gg 1`,
   :math:`\omega_I\tau \ll 1`) the dipolar spectral densities reduce to
   :math:`3\tau` for :math:`R_1` and :math:`4\tau+3\tau=7\tau` for
   :math:`R_2`, so :math:`R_2/R_1 \to 7/6`.

.. _EULER:

Susceptibility tensor and Euler angle convention
-------------------------------------------------

The magnetic susceptibility tensor :math:`\boldsymbol{\chi}` is decomposed
into an isotropic part and a traceless anisotropic (deviatoric) part:

.. math::

    \boldsymbol{\chi} = \chi_{\mathrm{iso}} \mathbf{I} + \Delta\boldsymbol{\chi}

where :math:`\chi_{\mathrm{iso}} = \tfrac{1}{3}\operatorname{tr}(\boldsymbol{\chi})`.

The anisotropic part is characterised by two invariants:

* **Axiality** :math:`\Delta\chi_{\mathrm{ax}}` — the largest principal deviation from isotropy.
* **Rhombicity** :math:`\Delta\chi_{\mathrm{rh}}` — the in-plane asymmetry.

**Euler angles** describe the passive ZYZ rotation that maps the input
(molecular) frame to the eigenframe of :math:`\boldsymbol{\chi}`.  The three
angles :math:`(\alpha, \beta, \gamma)` are defined as:

.. math::

    \alpha &= \operatorname{arctan2}(R_{31},\,-R_{11}) \\
    \beta  &= \arccos(R_{21}) \\
    \gamma &= \operatorname{arctan2}(-R_{23},\, R_{21})

where :math:`R_{ij}` are elements of the rotation matrix whose columns are
the eigenvectors of :math:`\boldsymbol{\chi}`, sorted in ascending order of
deviation from :math:`\chi_{\mathrm{iso}}`.  All angles are reported in
degrees in the range :math:`[0°,\,180°]` for :math:`\beta` and
:math:`(-180°,\,180°]` for :math:`\alpha` and :math:`\gamma`.

.. note::

   The eigenvector ordering places the eigenvector with the *smallest*
   deviation from :math:`\chi_{\mathrm{iso}}` first (index 0) and the one
   with the *largest* deviation last (index 2).  The axial direction therefore
   corresponds to the last eigenvector, and the angles describe how to align
   the molecular-frame Y-axis with the axial eigenvector.

References
^^^^^^^^^^

.. [Lang2020] Lang, L.; Ravera, E.; Parigi, G.; Luchinat, C.; Neese, F.
   J. Phys. Chem. Lett. 2020, 11 (20), 8735-8744.
   DOI: 10.1021/acs.jpclett.0c02462