.. _whats_new:

What's New
==========

This page summarises the major features and improvements introduced in the
current development branch relative to the last stable release.

----

Conformer ensemble averaging
-----------------------------

A new ``average_conformers`` CLI command computes population-weighted averages
of A tensors and ⟨r⁻⁶⟩ over a set of conformers from quantum-chemistry output
files, and writes a canonical molecule CSV suitable for use as a
``method: csv`` hyperfine input.

.. code-block:: bash

    simpnmr average_conformers conf1.out conf2.out conf3.out \
        --centre Fe1 \
        --weights 0.6 0.3 0.1 \
        --output averaged.csv

Weights are optional (defaults to equal weights) and need not be normalised.
All conformer files must have the same atom order. The output CSV is used
directly in any ``predict`` or ``fit_susc`` workflow via::

    hyperfine:
      method: csv
      file: averaged.csv

See :ref:`standalone_cli` for the full option reference.

----

New susceptibility methods for ``predict``
-------------------------------------------

Three additional susceptibility methods are available in the ``susceptibility``
block of the predict configuration.

**Spin-Hamiltonian (``sh``)**

Compute the susceptibility analytically from a full spin-Hamiltonian
parameterisation: principal g values, axial and rhombic ZFS parameters, and
the ZYZ Euler angles relating the g / ZFS eigenframe to the molecular frame.

.. code-block:: yaml

    susceptibility:
      method: sh
      sh:
        gx: 2.00
        gy: 2.10
        gz: 2.40
        D: 15.0          # cm⁻¹
        E_over_D: 0.10   # must be in [0, 1/3]
        alpha: 0.0
        beta: 30.0       # ZYZ Euler angles (degrees)
        gamma: 0.0

**Reduced-chiT (``reduced_chi``)**

Supply the three irreducible components of chiT directly (iso, ax, rh/ax
ratio) with Euler angles.  The full tensor is reconstructed from the Curie
prefactor.

**Bleaney (``bleaney``)**

Specify Stevens B²₀ and B²₂ parameters together with an isotropic g_J;
requires ``total_momentum_J`` to be set (lanthanide / actinide J-multiplets).

.. code-block:: yaml

    susceptibility:
      method: bleaney
      bleaney:
        B20: -0.12    # cm⁻¹
        B22: 0.04     # cm⁻¹
        g_J: 1.333

**J-multiplet support throughout**

All susceptibility and VT fitting code now uses J(J+1) and (2J−1)(2J+3)
when ``total_momentum_J`` is set, correctly describing lanthanide/actinide
systems.

----

g-corrected isotropic susceptibility
--------------------------------------

The VT susceptibility fitting now uses a fully g-corrected isotropic
susceptibility formula for the analytic reference component:

.. math::

   \chi_\text{iso} T \propto g_e g_\text{iso}
     - \frac{f(S)}{45 k_B T}\!\left(D\,g_e g_\text{ax} + 3E\,g_e g_\text{rh}\right)

where the products :math:`g_e g_\text{iso}`, :math:`g_e g_\text{ax}`, and
:math:`g_e g_\text{rh}` are computed directly from the g-tensor (frame-independent
invariants), matching the observable from an Evans-method measurement.

A frame-alignment check warns when the g-tensor and susceptibility eigenframes
are significantly misaligned.

----

Automatic τ\ :sub:`R` and rotational correlation time tools
--------------------------------------------------------------

**Ellipsoid hydrodynamic model (``calc_tau_c``)**

A new CLI command estimates τ\ :sub:`R` from molecular geometry using either
the Perrin ellipsoid or bead-shell hydrodynamic model with Arrhenius solvent
viscosity correction:

.. code-block:: bash

    simpnmr calc_tau_c structure.xyz --solvent water --method ellipsoid

**Auto-linewidth from structure**

When no relaxation model is configured in a ``predict`` run, τ\ :sub:`R` is
estimated automatically from the Perrin ellipsoid (Stokes–Einstein–Debye,
water at 298 K, B₀ = 11.75 T, τ\ :sub:`e` = 1 ps).

**Config keys for τ\ :sub:`R` per temperature**

In ``fit_relaxation``, set ``tau_r_method``, ``tau_r_solvent``, and related
keys to compute τ\ :sub:`R` automatically at each experimental temperature
instead of supplying a fixed value.

----

Per-isotope support
--------------------

Nuclei in a multi-isotope molecule (e.g. ¹H / ¹³C / ¹⁴N) are now handled
consistently throughout the pipeline:

- The ``chem_labels`` CSV accepts an optional ``isotope`` column to assign
  isotopes per nucleus (e.g. ``1H``, ``15N``).  Absent entries default to the
  element's natural NMR isotope.
- Prediction and fitting plots are generated separately per isotope.
- Support for ¹⁴N, ¹⁵N, and ¹⁹F added to the gamma / default-isotope tables.

----

r\ :sup:`−6` relaxation fitting and τ-space analysis
------------------------------------------------------

- Fit the distance-weighted sum of contributions to R\ :sub:`1` against
  experimental linewidths; CSV output of per-nucleus contributions.
- New τ-space plots show the accessible (τ\ :sub:`c`, τ\ :sub:`e`) space,
  with R\ :sub:`1` contours and τ\ :sub:`R` overlay lines.
- Combined τ-space overlay for all nuclei in a single figure.

----

Assignment improvements
------------------------

The Hungarian-algorithm assignment now incorporates:

- **Width cost** — penalises assignments whose predicted linewidths differ
  significantly from observed.
- **R\ :sub:`1` cost** — penalises assignments inconsistent with the observed
  R\ :sub:`1` relaxation rate.
- **Area-weighted** assignment uses peak integrals to improve assignments in
  crowded spectra.
- **Multi-exponential** support for overlapping peaks with a single chemical
  label.

----

Visualisation improvements
----------------------------

- **1σ confidence bands** on g\ :sub:`iso` solution-line plots and VT fits.
- **g\ :sub:`iso` Evans plot** — new panel showing ζ\ :sub:`eff` vs g\ :sub:`iso`
  with L-arrows for axial/rhombic contributions.
- **CSV export** of plotted data from shift, g\ :sub:`iso`, and spectrum figures.
- **Axis breaks** in spectrum plots for widely separated shift ranges.
- **Spectrum scaling** and dynamic peak labels proportional to linewidth.
- **Euler angle confidence intervals** reported with compact ± notation.
- **PCS isosurface colouring** — positive lobes red, negative lobes blue.

----

GUI improvements
-----------------

- **Unified predict / fit_susc GUI** (``simpnmr.gui.app``).
- **Embedded 3D molecule viewer** via 3Dmol.js / PyQt6-WebEngine.
- **Split fitter panel** with per-group Lorentzian controls.
- **Per-isotope shift plots** and a shared isotope / temperature selector.
- **Diamagnetic correction** controls with per-isotope reference shieldings.
- **Susceptibility method panels** show/hide dynamically based on the selected
  method (file, sh, bleaney, reduced_chi).

----

``label_groups`` CLI utility
------------------------------

A new standalone utility automatically assigns NMR-equivalent group labels
(methyl, *tert*-butyl) based on geometry:

.. code-block:: bash

    simpnmr label_groups structure.xyz --output labels.csv

----

Bug fixes
----------

- ORCA 6.1 output files now detected correctly (ASCII banner changed between
  6.0.x and 6.1.0; version-string fallback added).
- Atoms with no default NMR isotope (e.g. Ga, Co) no longer crash the
  molecule loader; they are assigned ``isotope = None`` and ignored by
  NMR-specific steps.
- PCS isosurface origin fixed: cube centred on the paramagnetic centre in
  the original molecular frame.
- τ-space combined plot: τ\ :sub:`R` overlay lines were transposed; corrected.
- ``fit_susc`` incomplete chemical labels now emit a warning rather than
  silently producing wrong assignments.
