.. _output_files:

Output files
============


All output files are written to the output directory defined in the ``project`` block of the input YAML file.

For susceptibility fitting workflows (``fit_susc``), the names of all generated files are printed to the terminal together with a short description. Depending on the selected options and workflow configuration, the following output files may be produced:


1. ``assigned_experiment_<TEMPERATURE>_K.csv``
   Assigned experimental data at a given temperature.

   Written in the wide-format CSV layout (see *Experiment CSV format* in
   :ref:`input_files`). The file is identical in structure to the input
   experimental CSV, with signal assignments updated to reflect the
   result of the fitting or assignment step.

2. ``dft_hyperfines.csv``  
   Raw hyperfine coupling constants extracted from a DFT calculation.

   Generated only when ``hyperfine: method: dft`` is selected.

3. ``hyperfines_and_shifts_<TEMPERATURE>_K.csv``  
   Combined hyperfine and shift data at a given temperature.

   Contains hyperfine coupling constants, chemical shifts, atomic coordinates, and chemical labels for each nucleus in the system.

4. ``pcs_isosurf_<TEMPERATURE>_K.cube``  
   Pseudocontact shift isosurface.

   Generated as a Gaussian cube file for visualisation of PCS fields in real space.

5. ``susceptibility_components_chi.pdf``  
   Temperature dependence of the magnetic susceptibility components.

   Shows :math:`\chi` versus :math:`T`.  
   Generated when more than one temperature is specified and ``--isoaxrho_plots on`` or ``--save`` is enabled.

6. ``susceptibility_components_chiT.pdf``  
   Temperature-scaled susceptibility components.

   Shows :math:`\chi T` versus :math:`T`.  
   Generated when more than one temperature is specified and ``--isoaxrho_plots on`` or ``--save`` is enabled.

7. ``susceptibility_tensor.csv``
   Fitted magnetic susceptibility tensors.

   Contains susceptibility tensors as a function of temperature, standard deviations of fitted parameters, goodness-of-fit metrics (:math:`r^2`, :math:`r^2_\mathrm{adj}`, MAE), eigenvalues of the susceptibility tensor, and an eigenvector representation using Euler angles.

8. ``shift_width_bubble_<TEMPERATURE>_K[_<ISOTOPE>].pdf``
   Scatter plot of observed chemical shift (x) versus linewidth (y) for each
   signal at a given temperature.

   Two panels are shown side by side:

   * **Matched** (left): signals whose assignment is present in both the
     experimental file and the molecule. Experimental markers are filled and
     scaled proportionally to the integrated peak area. When an r\ :sup:`−6`
     fit is available, predicted values are overlaid as open circles at the
     predicted shift position, connected to the corresponding experimental
     point by a dashed line.

   * **Unmatched** (right): signals present in only one dataset, drawn in a
     distinct highlight colour. Experimental unmatched markers are area-scaled
     (same convention as the matched panel). Predicted unmatched markers (open
     circles) are scaled by the number of equivalent nuclei sharing that
     chemical label (group size).

   Assignment labels are annotated next to every marker.

   Generated when relaxation fitting is enabled or experimental linewidth data
   are present. A per-isotope suffix is appended to the file name when the
   molecule contains more than one nuclear isotope.

9. ``shift_r1_bubble_<TEMPERATURE>_K[_<ISOTOPE>].pdf``
   Identical layout to ``shift_width_bubble`` but with longitudinal relaxation
   rate R\ :sub:`1` (s\ :sup:`−1`) on the y-axis.

   Generated only when R\ :sub:`1` data are present in the experimental file
   or an R\ :sub:`1` r\ :sup:`−6` fit has been performed.

10. ``r6_fit_width[_<ISOTOPE>]_<TEMPERATURE>_K.pdf`` / ``r6_fit_r1[_<ISOTOPE>]_<TEMPERATURE>_K.pdf``
    Observed versus fitted r\ :sup:`−6` distance model scatter plots.

    Experimental values (linewidth or R\ :sub:`1`) are plotted against
    r\ :sup:`−6` for each signal. The smooth fitted curve
    :math:`p_1 r^{-6} + p_2` is overlaid. Fitted parameters
    :math:`p_1`, :math:`p_2`, and RMSE are shown in an annotation box.

11. ``r6_tau_space_width[_<ISOTOPE>]_<TEMPERATURE>_K.pdf`` / ``r6_tau_space_r1[_<ISOTOPE>]_<TEMPERATURE>_K.pdf``
    τ parameter space plot for a single observable and temperature.

    Shows the (τ\ :sub:`e`, τ\ :sub:`R`) plane as a colour-mapped heatmap of
    :math:`\log_{10}(p_1^\mathrm{calc} / p_1^\mathrm{fit})`. The contour where
    :math:`p_1^\mathrm{calc} = p_1^\mathrm{fit}` (exact match) is drawn as a
    solid black line. Dashed lines mark the boundary of the bootstrap
    confidence interval (default 95%). Both axes are logarithmic and labelled
    in auto-selected units (fs, ps, ns, or µs).

    The relaxation model (``sbm``, ``curie``, or ``sbm curie``), temperature,
    and fitted :math:`p_1` value are shown in an annotation box.

12. ``r6_tau_space_combined[_<ISOTOPE>]_<TEMPERATURE>_K.pdf``
    Overlay of linewidth and R\ :sub:`1` τ-space constraints on one plot.

    The exact-match contour for each observable is drawn as a solid line in a
    distinct colour (primary palette for R\ :sub:`1`, highlight colour for
    linewidth). Dashed lines of the same colour show the bootstrap confidence
    interval boundaries. The region where both observables are simultaneously
    consistent with their fitted values is where the two contours intersect.

    Generated only when both linewidth and R\ :sub:`1` fits are available.

13. ``r6_tau_space_multitemp_width[_<ISOTOPE>].pdf`` / ``r6_tau_space_multitemp_r1[_<ISOTOPE>].pdf``
    Multi-temperature τ-space plot for a single observable.

    Each temperature produces one constraint contour (solid line) and a shaded
    confidence band, coloured on a sequential palette from low to high
    temperature. The intersection of all contours identifies the (τ\ :sub:`e`,
    τ\ :sub:`R`) pair consistent with every experimental temperature
    simultaneously.

    Generated when relaxation data are available at more than one temperature.
