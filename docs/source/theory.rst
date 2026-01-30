.. _theory:
Theory
======

Below we outline the key models and equations implemented in ``simpnmr``.

.. _PCS:

Pseudocontact shift
-------------------
Pseudocontact shift (PCS) depends on the anisotropic part of the magnetic susceptibility tensor and
the dipolar part of the hyperfine coupling tensor (HFC). Depending on the sourse of the HFC 
(output file form density functional theory (DFT) calculation or simply a file with coordinates) there 
are three approximations that are implemented in ``simpnmr``

1. PCS with point-dipole approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming that a paramagnetic metal centre is at the origin and a nucleus of interest has coordinates (x,y,z), 
the isotropic part of the pseudocontact shift (PCS) tensor :math:`\delta_{\mathrm{PCS}}` can be calculated as 
a third of the trace of the magnetic susceptibility tensor :math:`\chi` multiplied by the reduced dipolar hyperfine tensor,
which in the point-dipole approximation is a matrix that depends just on nuclear coordinates.

.. math::
   :label: :eq: pcs_pd

    \delta_{\mathrm{PCS}}=\frac{1}{12 \pi r^5} \operatorname{tr}\left[\left(\begin{array}{ccc}
    \chi_{x x} & \chi_{x y} & \chi_{x z} \\
    \chi_{y x} & \chi_{y y} & \chi_{y z} \\
    \chi_{z x} & \chi_{z y} & \chi_{z z}
    \end{array}\right) \cdot\left(\begin{array}{ccc}
    3 x^2-r^2 & 3 x y & 3 x z \\
    3 x y & 3 y^2-r^2 & 3 y z \\
    3 x z & 3 y z & 3 z^2-r^2
    \end{array}\right)\right]

If the coordinates are specified in Å, and the :math:`\chi` is in Å\ :sup:`3` , then the equation above multiplied by 10\ :sup:`6` 
gives the PCS in ppm.


2. PCS with non-relativistic hyperfine from DFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To account for the non-point nature of the paramagnetic center, the normaised reduced dipolar hyperfine can be taken from a simple 
single-point non-relativistic DFT calculation :math:`\mathbf{A}^{\mathrm{dip}}`.

.. math::
   :label: :eq: pcs

   \delta_{\mathrm{PCS}}=\frac{1}{3} \operatorname{tr}\left(\Delta \boldsymbol{\chi} \cdot \mathbf{A}^{\mathrm{dip}}\right)

Note, that since the dipolar hyperfine is traceless, the isotropic part of PCS does not depend on the isotropic part of the magnetic susceptibility tensor. 
Hence the equation features the traceless :math:`\Delta \boldsymbol{\chi}`.

3. PCS with relativistic hyperfine from DFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If relativistic contribution to HFC :math:`\mathbf{A}^{\text {orb }}` and the g-tensor are calculated in the same DFT output, 
then an additional contribution to the PCS coming from :math:`\mathbf{A}^{\text {orb }}` can be evaluated.

.. math::
   :label: :eq: pcs_orb

   \delta_{\mathrm{PCS}}^{\text {orb }}=\frac{1}{3} \operatorname{tr}\left(\Delta \boldsymbol{\chi} \cdot\left(\frac{\mathrm{~g}_{\mathrm{e}}}{\mathbf{g}^{\mathrm{T}}} \cdot\left(\mathbf{A}^{\operatorname{dip}}+\mathbf{A}^{\text {orb }}\right)^{\mathrm{T}}-\mathbf{A}^{\operatorname{dip}}\right)\right)

This contribution should go to zero with the distance from paramagnetic center for light nuclei.

.. _FC:

Fermi-contact shift
-------------------

Fermi-contact shift often described as the contribution arisig form the spin density at the nucle

FC with spin-only magnetic susceptibility
-----------------------------------------

In the simplest model, Fermi-contact shift is proportional to the reduced Fermi-contact hyperfine at the nucleus of interest and the spin-only magnetic susceptibility value 

.. math::
   :label: :eq: fc_s

    \delta_{\mathrm{FC}}=\chi_{iso}^S A^{FC}

where the spin-only magnetic susceptibility in SI units is 

.. math::
   :label: :eq: chi_s

    \chi_{iso}^S=\frac{\mu_0 \mu_B^2 \mathrm{g}_{\mathrm{e}}^2 S(S+1)}{3 k T}

where :math:`\mu_0` is the vacuum permeability, :math:`\mu_B` is Bohr magneton, :math:`\mathrm{g}_{\mathrm{e}}` is a free elecrton g-factor,
:math:`S` is the total spin, :math:`k` is the Boltzmann constant and :math:`T` is the temperature.

Accounting for g-tensor anisotropy 
----------------------------------

In order to account for the effect of the g-tensor anisotropy on the FC shift we need both magnetic susceptibility tensor 
and the g-tensor calculated at the same level of theory e.g. SOC-NEVPT2

.. math::
    :label: :eq: FC_g

    \delta^{F C}=\frac{\mathrm{g}_{\mathrm{e}}}{3} \operatorname{tr}\left(\frac{\boldsymbol{\chi}}{\mathbf{g}^{\mathrm{T}}}\right) A^{FC}  


Relativistic hyperfine from DFT
-------------------------------

If relativistic hyperfine contribution :math:`\mathbf{A}^{\text {orb }}` and the g-tensor are calculated with a DFT method in ORCA, additional contribution to the shift,
which does not depend on the spin-only HFC :math:`A^{FC}` but on the isotropic magnetic susceptibility tensor can be evaluated. 

.. math::
    :label: :eq: FC_orb

    \delta^{\text {orb }}=\chi_{\text {iso }} \frac{1}{3} \operatorname{tr}\left(\frac{\mathrm{~g}_{\mathrm{e}}}{\mathbf{g}^{\mathrm{T}}} \cdot\left(\mathbf{A}^{\text {dip }}+\mathbf{A}^{\text {orb }}\right)^{\mathrm{T}}\right)

as with another term arising from relativistic contribution to the hyperfine tensor, the term above vanished at large distances from the paramagnetic center.

.. _PRE:

Paramagnetic relaxation enhancement
===================================

