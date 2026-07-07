# CHANGELOG

<!-- version list -->

## v2.1.1 (2026-07-07)

### Bug Fixes

- **exp**: Read a trailing isotope column in wide-format experiment files
  ([`74014a2`](https://gitlab.com/suturina-group/simpnmr/-/commit/74014a2d1e453e6d553b9310ffd20d76ba247887))

- **exp**: Tag comment-format experiment signals with the isotope header
  ([`e9f669d`](https://gitlab.com/suturina-group/simpnmr/-/commit/e9f669d7c1046689df50e28a6af311a9885f808d))

- **gui**: Drop the isotope/temperature title from spectrum panels
  ([`c24b3ad`](https://gitlab.com/suturina-group/simpnmr/-/commit/c24b3ad8aeb6a92e572f765008dd5a8d97cb7a86))

- **gui**: Isolate post-run result loaders so a display error can't crash the app
  ([`af15a51`](https://gitlab.com/suturina-group/simpnmr/-/commit/af15a51a62e8d03eaf5fd49dea13b4eceff29bc7))

- **packaging**: Bundle matplotlib output backends in Windows build
  ([`e7c0010`](https://gitlab.com/suturina-group/simpnmr/-/commit/e7c0010089b7160a682a647f000f13082d3a9228))

- **predict**: Warn on susceptibility/hyperfine geometry mismatch
  ([`4f1b93b`](https://gitlab.com/suturina-group/simpnmr/-/commit/4f1b93bc49ecaea50dbbba2fd14c3010cec98ace))

- **security**: Harden config parsing, viewer HTML, and output paths
  ([`545eee5`](https://gitlab.com/suturina-group/simpnmr/-/commit/545eee57b792c94db00bd89445650614cdf5bd71))

### Continuous Integration

- **windows**: Also attach the installer to the GitLab release on manual runs
  ([`d293e5c`](https://gitlab.com/suturina-group/simpnmr/-/commit/d293e5c7cdbda71d7b987fc556bf13e5fb4840f0))

- **windows**: Bump actions to Node 24 majors
  ([`eb82fc2`](https://gitlab.com/suturina-group/simpnmr/-/commit/eb82fc223c3f9e0bdeb074a6b4e812209924e9cc))

- **windows**: Fix PowerShell parse error from non-ASCII in build scripts
  ([`fed9a1e`](https://gitlab.com/suturina-group/simpnmr/-/commit/fed9a1ea3fc5844a3869b9af320f490910882f50))

### Documentation

- Add a license and liability disclaimer to the home page
  ([`229efb2`](https://gitlab.com/suturina-group/simpnmr/-/commit/229efb2ccda74912a0d993c1ecf8f13bb691ba18))

- Add the SimpNMR preprint (ChemRxiv DOI) to the citation notes
  ([`404192b`](https://gitlab.com/suturina-group/simpnmr/-/commit/404192b211a902b302cb30936f4b8052ba76c24b))

- Link the Windows installer directly (auto-versioned)
  ([`3ebae6e`](https://gitlab.com/suturina-group/simpnmr/-/commit/3ebae6efd74017297b3322308bdad38e691e7df4))


## v2.1.0 (2026-07-06)

### Bug Fixes

- **predict**: Only model relaxation when a relaxation block is configured
  ([`49e24c7`](https://gitlab.com/suturina-group/simpnmr/-/commit/49e24c7ac08fd6483a604e235276bcd14335218f))

- **susc**: Supply g_e·g cross-products in Bleaney susceptibility
  ([`9bc3d45`](https://gitlab.com/suturina-group/simpnmr/-/commit/9bc3d45c1e5a56e13d4904fd448c1d370d7e6c9b))

- **susc**: Use Curie g_J² for the Bleaney isotropic susceptibility
  ([`d43afe5`](https://gitlab.com/suturina-group/simpnmr/-/commit/d43afe544cabed09e85ad19e1c8c2fda8879b312))

- **viz**: Round broken-axis spectrum fallback ticks
  ([`6b7327e`](https://gitlab.com/suturina-group/simpnmr/-/commit/6b7327eee46e42b878c8069b64606beac33e7c38))

### Documentation

- Add macOS Automator app instructions
  ([`707e3bb`](https://gitlab.com/suturina-group/simpnmr/-/commit/707e3bb003d9c7b02e0d37d79e36ad535e585bf8))

- Add T, frequency and solvent to the experimental spectrum caption
  ([`8ba98d7`](https://gitlab.com/suturina-group/simpnmr/-/commit/8ba98d731d076775c3ee93dcf5314ffb1926faff))

- Beginner-friendly, OS-tabbed installation guide
  ([`c97bcf8`](https://gitlab.com/suturina-group/simpnmr/-/commit/c97bcf85c9f4b322fc59969625eddf75015df376))

- Correct deconvolution wording (spectrum -> sum of Lorentzians)
  ([`95ed2c8`](https://gitlab.com/suturina-group/simpnmr/-/commit/95ed2c835b12887362d95a46983a3618cac10818))

- Describe B20 as the Stevens-operator crystal-field parameter
  ([`cc8788a`](https://gitlab.com/suturina-group/simpnmr/-/commit/cc8788ab4b96cf30030dcc74bcd2fcfa0977f5d4))

- Embed the predicted spectrum figure in the Dy tutorial
  ([`c13bd42`](https://gitlab.com/suturina-group/simpnmr/-/commit/c13bd4240800157e3028c05d6788fe9eb7499324))

- Embed the relaxation spectrum figure in the Dy tutorial
  ([`37b6e1b`](https://gitlab.com/suturina-group/simpnmr/-/commit/37b6e1b40881ff7c229913b7ae3610ef715307c4))

- Expand and correct the Dy susceptibility-fitting tutorial
  ([`ef2cc34`](https://gitlab.com/suturina-group/simpnmr/-/commit/ef2cc34cfb5dbe0eded481eccfa7fccc5fd6809c))

- Explain peak deconvolution and experiment.csv columns in Dy fit
  ([`c81c0a7`](https://gitlab.com/suturina-group/simpnmr/-/commit/c81c0a7003a34bc595e70855b0458388d2a286a5))

- Give each tutorial workflow a descriptive project name
  ([`f88087d`](https://gitlab.com/suturina-group/simpnmr/-/commit/f88087dab6cb8e645a97858f737ee1b63d3fb770))

- Inline tutorial example files, drop bulk examples.zip
  ([`8538713`](https://gitlab.com/suturina-group/simpnmr/-/commit/8538713e1c1a491f81b15beed85a4c08e659a476))

- Open the Dy fitting section with the experimental spectrum
  ([`89e13fa`](https://gitlab.com/suturina-group/simpnmr/-/commit/89e13fa83b68e6b62c1a2b7c286cfce8de5e19cb))

- Rename Dy fitting section to cover assignment and susceptibility fit
  ([`83b338c`](https://gitlab.com/suturina-group/simpnmr/-/commit/83b338c60ec7a166c42edc9373181d5321d4e057))

- Render the experiment.csv peak data as a table in the Dy fit
  ([`324db82`](https://gitlab.com/suturina-group/simpnmr/-/commit/324db829bb6ec8fbc1601c9435d364048ed95a51))

- Restructure tutorials into Dy(III) and Fe(II) worked examples
  ([`964c5b8`](https://gitlab.com/suturina-group/simpnmr/-/commit/964c5b81a69e244cc16f607ec4b4f63b1d2aa94c))

- Show unmeasured R1 as empty, not 0, in the Dy fit example
  ([`5935e2c`](https://gitlab.com/suturina-group/simpnmr/-/commit/5935e2cac0f180ea32dbca67f4091aa229f7fce8))

- Use B20 = -100 cm^-1 in the Dy tutorial and regenerate spectra
  ([`b295dc8`](https://gitlab.com/suturina-group/simpnmr/-/commit/b295dc8a8ad98b91d902b28da1fd4931bb700766))

- Use methanol as the τ_R solvent in the Dy tutorial
  ([`a4da8af`](https://gitlab.com/suturina-group/simpnmr/-/commit/a4da8af69d303c75ada7a6d6ec22674904d86e56))

### Features

- **packaging**: Windows desktop installer with GitLab release upload
  ([`e27bb38`](https://gitlab.com/suturina-group/simpnmr/-/commit/e27bb38c6c990cef1df78fc61324447693597821))

- **relaxation**: Estimate tau_R from solvent and temperature in predict
  ([`744c363`](https://gitlab.com/suturina-group/simpnmr/-/commit/744c3636362b6cb8b7816b2c8a0314694d3b404e))


## v2.0.0 (2026-07-03)

### Bug Fixes

- **config**: Add include_groups to PredictConfig nuclei KEYWORDS
  ([`2207d78`](https://gitlab.com/suturina-group/simpnmr/-/commit/2207d78446a7033b597338bad7c7cd3a0b4dd3ad))

- **dia**: Per-isotope reference shieldings for DFT diamagnetic correction
  ([`46f251d`](https://gitlab.com/suturina-group/simpnmr/-/commit/46f251d07fb062bbdc4accd842d8195be225aba2))

- **gui**: Bundle 3Dmol.js to fix blank viewer in QWebEngine
  ([`a4c9ef7`](https://gitlab.com/suturina-group/simpnmr/-/commit/a4c9ef79318ec312a3b7465799a9c581c4b8bfbe))

- **viz**: Use predicted shifts in bubble plot connectors, fix shift_spread KeyError, show reduced
  chi in fitted shifts
  ([`99f9c74`](https://gitlab.com/suturina-group/simpnmr/-/commit/99f9c74d5ae4ec0048d2775e5aea85008aa6c6dd))

- **vt**: Correct NameError in VT ab-initio plot and refresh stale test fixtures
  ([`f2085d4`](https://gitlab.com/suturina-group/simpnmr/-/commit/f2085d4e2640935d5a37ae8b6e9c11342dffa61f))

- **vt**: Restore example test configs and fix latent isoaxrh/Bleaney bugs
  ([`8cda847`](https://gitlab.com/suturina-group/simpnmr/-/commit/8cda847790ec821d532f158a79117f1c6cc9c059))

### Documentation

- Add getting-started tutorial and fix CLI reference/underlines
  ([`d02a24e`](https://gitlab.com/suturina-group/simpnmr/-/commit/d02a24e3e61d5635db037e33f4393f3a4ce7f604))

### Features

- Broken-axis spectra, R1/R2 relaxation decomposition, peak-data files, correlation-driven
  permutations
  ([`13fdb1a`](https://gitlab.com/suturina-group/simpnmr/-/commit/13fdb1afbbed78b9bbca7010e0454d5ba9f28e2e))

- Calc_tau_c CLI, ellipsoid τR auto-computation, combined τ-space overlay
  ([`2fc166b`](https://gitlab.com/suturina-group/simpnmr/-/commit/2fc166b55710cec3923e49d55acd654e25155fe9))

- Conformer averaging, g_iso Evans, VT TIP fix, 17O, susc plot polish
  ([`f934d00`](https://gitlab.com/suturina-group/simpnmr/-/commit/f934d00878fb3b681a1903ad6a1eec5d2431f451))

- Dynamic spectrum labels, min_linewidth_hz, __contains__ fix, and spectrum scaling
  ([`cca8f2a`](https://gitlab.com/suturina-group/simpnmr/-/commit/cca8f2ae8fc5113255888d41cefd1d097bb5ba0c))

- Isotope-aware assignment, wide CSV format, multi-exp Hungarian
  ([`c2b7d5b`](https://gitlab.com/suturina-group/simpnmr/-/commit/c2b7d5bd81da4502f50435bfdc1710b61aa8d127))

- Multi-isotope pipeline fixes, conformer averaging, GUI assignment, docs
  ([`2021a26`](https://gitlab.com/suturina-group/simpnmr/-/commit/2021a2622f1f229fa8c1e8fc9f1e97435014103c))

- Per-isotope plots, tBu label fixes, bubble plot area scaling
  ([`58c29af`](https://gitlab.com/suturina-group/simpnmr/-/commit/58c29afdaf1b389496e54301cf2e743c50113f6e))

- R^-6 relaxation fit, area-weighted Hungarian assignment, and spectrum improvements
  ([`51ed7e5`](https://gitlab.com/suturina-group/simpnmr/-/commit/51ed7e56ee85cfd3637a62a4c077c72344469f9a))

- **assignment**: Add width and R1 cost terms to Hungarian assignment
  ([`d1727c6`](https://gitlab.com/suturina-group/simpnmr/-/commit/d1727c6df68bcb53ff4a98323588457a81a406db))

- **assignment**: HMBC/HSQC correlation constraints for permutation search
  ([`8e02158`](https://gitlab.com/suturina-group/simpnmr/-/commit/8e02158b47c74a5d5d5797a05d7b5783695d6ff6))

- **assignment**: Shared permutation assignment across temperatures
  ([`c8f8549`](https://gitlab.com/suturina-group/simpnmr/-/commit/c8f8549b01f15ec7ea93d5db4045d43d9b3aaea5))

- **gui**: Add diamagnetic method and per-isotope reference controls
  ([`5c56f59`](https://gitlab.com/suturina-group/simpnmr/-/commit/5c56f593da896c22170a523ae93f3fd0bfbaa60a))

- **gui**: Embed 3D molecule viewer using 3Dmol.js / PyQt6-WebEngine
  ([`0448a7b`](https://gitlab.com/suturina-group/simpnmr/-/commit/0448a7bdb898440f35e9c87e665b6ba709da4d93))

- **gui**: Shared iso/temp selector, per-isotope shift plots, GUI fixes
  ([`b4174a3`](https://gitlab.com/suturina-group/simpnmr/-/commit/b4174a3746e83b59c1ea688e268dafea5f79b8a1))

- **gui**: Split fitter, per-group Lorentzians, 3D label/color polish
  ([`94cd1bf`](https://gitlab.com/suturina-group/simpnmr/-/commit/94cd1bf2c1324b92db3284bb5a0469a940f8df9a))

- **isotopes**: Per-nucleus isotope from chem_labels CSV, remove exp.isotope, add 14N/15N/19F
  ([`b66f6d5`](https://gitlab.com/suturina-group/simpnmr/-/commit/b66f6d51c92fcb3c15b73fb498f3a59d7db45016))

- **predict**: Add R1 and linewidth decomposition plots; fix label rotation
  ([`a14e91b`](https://gitlab.com/suturina-group/simpnmr/-/commit/a14e91ba343dc4451536a04ff438f58c24e2ae5d))

- **predict**: Add sh and reduced_chi susceptibility methods; fix PCS centering
  ([`752a765`](https://gitlab.com/suturina-group/simpnmr/-/commit/752a765f4437d12f491b4e992efabc8ef34e07a9))

- **predict**: Auto τ_r linewidth, atom-label centres, Bleaney susc, J-multiplet χ(T)
  ([`6d01233`](https://gitlab.com/suturina-group/simpnmr/-/commit/6d01233903923b9536c35eb91758675bb47a7276))

- **predict**: Spin-only susceptibility mode, VT spectra plot, PNG export
  ([`4af419c`](https://gitlab.com/suturina-group/simpnmr/-/commit/4af419c6bcc0e1cda0143a9e6078e98beba90084))

- **relaxation**: Τ-space plots and r^-6 fit CSV output
  ([`a4b7986`](https://gitlab.com/suturina-group/simpnmr/-/commit/a4b79865be7a13d6cb5c8d4471574056464f5c39))

- **tools**: Add label_groups CLI for methyl/tBu group labelling
  ([`e69df3b`](https://gitlab.com/suturina-group/simpnmr/-/commit/e69df3bd328a3a20026da7b0af11d1397920cca7))

- **viz**: 1σ bands, CSV export, robust x-axis for g_iso solution plot
  ([`63b94bc`](https://gitlab.com/suturina-group/simpnmr/-/commit/63b94bc38c3ddc70a114378468d999851e9853d6))

- **viz**: Axis breaks, panel height matching, and spectrum polish
  ([`0be51fc`](https://gitlab.com/suturina-group/simpnmr/-/commit/0be51fc652ed59f988dba6b56cdf5e0efe766727))

- **viz**: Euler angle CIs, compact uncertainty, figure polish
  ([`f76fba4`](https://gitlab.com/suturina-group/simpnmr/-/commit/f76fba4622471624562eb9899e7ae105263d06b3))

- **viz**: G_iso solution-line plot with ζ panel and error propagation
  ([`8e39af2`](https://gitlab.com/suturina-group/simpnmr/-/commit/8e39af2ab6e41cfca69997f059d6bcb411c81075))

- **viz**: Per-isotope r^-6 fits and spectrum plots, fix contour clipping
  ([`f92cf49`](https://gitlab.com/suturina-group/simpnmr/-/commit/f92cf4964626d2dd5e617743d8b51ab845afd353))

- **viz**: Spectra figure polish, tau-ps units, GUI tau range controls
  ([`2775b48`](https://gitlab.com/suturina-group/simpnmr/-/commit/2775b4873f675c0375266789f21b5367bc865ae5))

### Performance Improvements

- **assignment**: Constrained permutation generation via backtracking
  ([`ac7bdac`](https://gitlab.com/suturina-group/simpnmr/-/commit/ac7bdac27f8a90f200fcaee5ea38a55f333b23de))

### Refactoring

- Rename rho → rh throughout (variables, keys, filenames, labels)
  ([`e912c95`](https://gitlab.com/suturina-group/simpnmr/-/commit/e912c951ac17d3766cd964f1b941066c9c88277b))

- **gui**: Unified predict/fit_susc GUI, rename to simpnmr.gui.app
  ([`1f790a2`](https://gitlab.com/suturina-group/simpnmr/-/commit/1f790a29ef296a3b9cf792f018143fb1ac9348da))

### Testing

- **spinham**: Fix stale rho -> rh type in get_sh fixture CSV
  ([`5fa9c74`](https://gitlab.com/suturina-group/simpnmr/-/commit/5fa9c74d250387971b037185451f80ecbed1eaf8))

### Breaking Changes

- Peak-data filenames now embed the magnetic field (peak_data_<T>_K_<B>_T.csv); the R2_total peak
  column is replaced by its components (R2_sbm_dipolar/contact and R2_curie); the default CSV float
  format changed from %.6f to %.10g; and fit_susc spectrum linewidths are taken from the relaxation
  model when tau_R and an r^-6 fit are available.


## v1.9.1 (2026-05-05)

### Bug Fixes

- **setup**: Render README on PyPI project page
  ([`ed27db0`](https://gitlab.com/suturina-group/simpnmr/-/commit/ed27db08c609131f58857a0f6cefc4cdb93a88ef))


## v1.9.0 (2026-05-05)

### Bug Fixes

- **app**: Validate point-dipole HFC inputs
  ([`6933797`](https://gitlab.com/suturina-group/simpnmr/-/commit/6933797ed5f39ff31585d8d45da231793fd9ec1b))

- **app**: Warn about incomplete chemical labels
  ([`eac8f85`](https://gitlab.com/suturina-group/simpnmr/-/commit/eac8f85a7e548fba9355ba1bcf287b3cb63182e6))

- **core**: Remove implicit default linewidth
  ([`2115dcf`](https://gitlab.com/suturina-group/simpnmr/-/commit/2115dcfb01670d2e46beb6cf47735b158af7ae95))

### Features

- **app**: Resolve automatic output linewidths
  ([`bde17ee`](https://gitlab.com/suturina-group/simpnmr/-/commit/bde17ee536820fcf0e56b09c49c99139d8068dc1))

- **io**: Label peak linewidth output mode
  ([`691159c`](https://gitlab.com/suturina-group/simpnmr/-/commit/691159c6b20806fb836a9075323a0a6e35d17454))

- **viz**: Use resolved linewidths for spectra
  ([`bc5a2e2`](https://gitlab.com/suturina-group/simpnmr/-/commit/bc5a2e206ff1accec899359b3f87bfb5d28a0a28))


## v1.8.4 (2026-04-29)

### Bug Fixes

- **vt**: Use analytic iso from ab initio g components
  ([`5cd77ab`](https://gitlab.com/suturina-group/simpnmr/-/commit/5cd77ab0fd022bdf08ed2ef4527816fd17b290c2))


## v1.8.3 (2026-04-23)

### Bug Fixes

- Guard axial-only g solve against unphysical radicand
  ([`cfceae8`](https://gitlab.com/suturina-group/simpnmr/-/commit/cfceae84f146dbc91b437cfdcc326b9eb3e2ae3d))


## v1.8.2 (2026-03-30)

### Bug Fixes

- Bump version to trigger package rebuild
  ([`a145fad`](https://gitlab.com/suturina-group/simpnmr/-/commit/a145fada2223c9d1dd6738d1333583dd3fd0abed))


## v1.8.1 (2026-03-29)

### Bug Fixes

- **viz**: Render TIP annotation with mathtext
  ([`ad7f0f0`](https://gitlab.com/suturina-group/simpnmr/-/commit/ad7f0f072fb7f721246d9ae7193eaee3f6259de3))


## v1.8.0 (2026-03-29)

### Chores

- **viz**: Reduce paper profile line width
  ([`cf3167c`](https://gitlab.com/suturina-group/simpnmr/-/commit/cf3167c48a6d0ce3149a83c6ce38641b5979702d))

- **viz**: Scale spectrum line width by 0.75
  ([`48cdc53`](https://gitlab.com/suturina-group/simpnmr/-/commit/48cdc532dcca614b82d4f36fe01986b6958611d4))

### Features

- **app**: Add susceptibility fit input units normalization
  ([`b3a710f`](https://gitlab.com/suturina-group/simpnmr/-/commit/b3a710f07ff2dd4d8cd3d3b2b2d4dad99d93b551))

### Refactoring

- **io**: Remove peak-level duplicate columns from molecule CSV export
  ([`91b3eac`](https://gitlab.com/suturina-group/simpnmr/-/commit/91b3eac79075459b4c12d327a783daf08f8c5f47))


## v1.7.2 (2026-03-28)

### Bug Fixes

- **predict,cfg,docs**: Add relaxation condition override policy with experiment fallback
  ([`047e3a0`](https://gitlab.com/suturina-group/simpnmr/-/commit/047e3a05318dd35ba5cecf2cbe4903f3bceeb069))


## v1.7.1 (2026-03-27)

### Bug Fixes

- **docs**: Update theory formulas and add references section
  ([`6a5358f`](https://gitlab.com/suturina-group/simpnmr/-/commit/6a5358f7c2bd923130c3a0fc983e4540cd7b9c38))


## v1.7.0 (2026-03-27)

### Bug Fixes

- **app**: Load optional DFT g-tensor in susceptibility fitting pipeline
  ([`ed87669`](https://gitlab.com/suturina-group/simpnmr/-/commit/ed876699e891505d1291db71b7beb41f94fe2f5b))

- **app**: Validate DFT g-tensor shape in loader
  ([`90ec5c6`](https://gitlab.com/suturina-group/simpnmr/-/commit/90ec5c60a386bec516a0a9067fec96ddd77c385c))

- **cfg**: Make susceptibility file optional in prediction config
  ([`6cf3a6b`](https://gitlab.com/suturina-group/simpnmr/-/commit/6cf3a6b34aea9f0027be3227659efb529834c2d8))

- **cfg**: Parse paramagnetic centre strings with yaml
  ([`1949a46`](https://gitlab.com/suturina-group/simpnmr/-/commit/1949a46ecaba5e7ea51fb162a38fbe8e9bc79cc9))

- **core**: Add csv susceptibility builder with spin-only iso
  ([`cbde566`](https://gitlab.com/suturina-group/simpnmr/-/commit/cbde5664b66f4d765e004e9e482bafa46bf4a999))

- **corr-time**: Match grouped experimental chem labels directly in R1 fit
  ([`1762296`](https://gitlab.com/suturina-group/simpnmr/-/commit/176229685e2964f761f6eb7cabbfb8a0f742987d))

- **domain**: Rotate paramagnetic centre with molecule frame
  ([`3133c80`](https://gitlab.com/suturina-group/simpnmr/-/commit/3133c8025996b8e5e84a13cfa84aadbc42e9e06b))

- **domain**: Store point-dipole HFC in canonical molecule state
  ([`2dcfc36`](https://gitlab.com/suturina-group/simpnmr/-/commit/2dcfc36bac790768b0ec86e197f2029659c8114f))

- **domain**: Update full hyperfine tensor after point-dipole accumulation
  ([`ee4f2c9`](https://gitlab.com/suturina-group/simpnmr/-/commit/ee4f2c9b5f8c00b10dbae4e6e3e937d8cf6e20d8))

- **fit**: Remove assignment-prefix heuristic from corr_time_fit experiment R1 collection
  ([`9e0bade`](https://gitlab.com/suturina-group/simpnmr/-/commit/9e0bade97ab4fa09701b2521ef340becce3331f7))

- **fitting**: Assign canonical susceptibility iso from fitted model
  ([`ee26f6f`](https://gitlab.com/suturina-group/simpnmr/-/commit/ee26f6f0e0764b1553d9a66d7a9bc46a143d5d6e))

- **io**: Accept R1 (Hz) headers when loading experiment CSVs
  ([`1b83daf`](https://gitlab.com/suturina-group/simpnmr/-/commit/1b83daf713815edd0a6aa39b3bdae4714cfcb18f))

- **io**: Allow Gaussian spin-dipole parsing with a single output block
  ([`10108c4`](https://gitlab.com/suturina-group/simpnmr/-/commit/10108c4f8e79c0f2af23ef4eb23197dc23fdc372))

- **io**: Map Gaussian hyperfine components to canonical QCA tensors
  ([`6f38956`](https://gitlab.com/suturina-group/simpnmr/-/commit/6f38956a45be944372745e97c02f58b2f259225a))

- **loader**: Validate paramagnetic centre against molecule geometry
  ([`7843ef4`](https://gitlab.com/suturina-group/simpnmr/-/commit/7843ef42b63e15b6f9771893a9f013c77ac96479))

- **loaders**: Make load_paramagnetic_centre work correctly in tau-fit pipeline
  ([`d5b7a95`](https://gitlab.com/suturina-group/simpnmr/-/commit/d5b7a9571cb97cd444c6ca1a8aae8424fac12c59))

- **susc**: Reduce loader log spam
  ([`7230eb5`](https://gitlab.com/suturina-group/simpnmr/-/commit/7230eb5fa2fdcbc1f5b5592af9dcb651e77e08da))

- **viz**: Format fitted Euler angles as integers
  ([`9ba1766`](https://gitlab.com/suturina-group/simpnmr/-/commit/9ba176655c64493bcb6ff2eaa888c47d7927fb7e))

- **viz**: Prefer earlier label offset tiers in layout scoring
  ([`ee75072`](https://gitlab.com/suturina-group/simpnmr/-/commit/ee750724a3ca02c0b12279e7291d91fd299f9fe4))

- **viz**: Preserve trailing zeros in compact uncertainty formatting
  ([`80f473f`](https://gitlab.com/suturina-group/simpnmr/-/commit/80f473f2c02d1cb13d5f1a095b7baa456f46b2cc))

### Chores

- **core**: Add todo for domain-derived relaxation evaluation context
  ([`b1a0838`](https://gitlab.com/suturina-group/simpnmr/-/commit/b1a0838811ee6d95c23d6ca22c02a46a0690f66b))

- **gitignore**: Ignore example simulation output directories
  ([`2b76ff6`](https://gitlab.com/suturina-group/simpnmr/-/commit/2b76ff614946a5286846e3968b278bb43dd4e434))

- **gitignore**: Ignore generated pcs isosurface test cubes
  ([`5fb04c5`](https://gitlab.com/suturina-group/simpnmr/-/commit/5fb04c5f8d886f10c6f40940d2a56c4673284364))

- **gitignore**: Ignore generated spinham test artefact
  ([`ab0635f`](https://gitlab.com/suturina-group/simpnmr/-/commit/ab0635fe902b96739e6204d97fe0c5224bba68bc))

- **gitignore**: Stop tracking example simulation outputs
  ([`c7e29ca`](https://gitlab.com/suturina-group/simpnmr/-/commit/c7e29ca36998ff44cf0f2e91fb98d489c61e6512))

- **logging**: Update peak CSV export log message
  ([`1528c46`](https://gitlab.com/suturina-group/simpnmr/-/commit/1528c4644a225bba9ea3378c0aeecac106b9aacd))

- **test**: Stop tracking generated spinham chiT artefact
  ([`85a9487`](https://gitlab.com/suturina-group/simpnmr/-/commit/85a94874a685b70353505d0125f70f93a60213cc))

### Code Style

- **corr-time**: Add import section comments
  ([`2bb3199`](https://gitlab.com/suturina-group/simpnmr/-/commit/2bb31994deb332263e1720028bbf4e79e20a871d))

- **fit**: Add delta notation to anisotropic iso-ax-rho model labels
  ([`32e180f`](https://gitlab.com/suturina-group/simpnmr/-/commit/32e180fd197fd10ac54b8135b5fd743e127d9a75))

- **fit**: Spell out chi_rho in iso-ax-rho display label
  ([`441212f`](https://gitlab.com/suturina-group/simpnmr/-/commit/441212f80edb90e2d6b6009fd2703c7d9883b9de))

- **glyphs**: Reduce paper-profile line widths
  ([`cb5be71`](https://gitlab.com/suturina-group/simpnmr/-/commit/cb5be71a3f93456a59ea473c77abb62a8b76d3b3))

- **viz**: Add delta notation to susceptibility comparison y-axis label
  ([`0b82c06`](https://gitlab.com/suturina-group/simpnmr/-/commit/0b82c066927136610e0ee11a0d1dae6fbb0c6584))

- **viz**: Change default compact uncertainty precision to one significant digit
  ([`34e2bcf`](https://gitlab.com/suturina-group/simpnmr/-/commit/34e2bcf7818b2369043273bf858e1d2d8e73eed2))

- **viz**: Darken corr-time scatter marker fill and refine marker outline
  ([`8945406`](https://gitlab.com/suturina-group/simpnmr/-/commit/8945406e171248fa9b45e317642babc2f3c8ca77))

- **viz**: Darken fitted-shift marker fill and soften marker outline
  ([`608e2e0`](https://gitlab.com/suturina-group/simpnmr/-/commit/608e2e0d2f023adf2d1f8378d7c8dec7f23bd4f4))

- **viz**: Reduce paper annotation font size from 8 to 7
  ([`e708bbb`](https://gitlab.com/suturina-group/simpnmr/-/commit/e708bbbd1ea0c72a3f70da78b561f5cf95290747))

- **viz**: Reduce spectrum connector line width
  ([`200fd61`](https://gitlab.com/suturina-group/simpnmr/-/commit/200fd615af0ef08d2a0add91c49e93b942ba8419))

- **viz**: Reduce stacked spectrum side-label font size by 1 pt
  ([`53b526d`](https://gitlab.com/suturina-group/simpnmr/-/commit/53b526d533a69d7998e9322ec515f72a06bbbf2d))

- **viz**: Refine corr-time plot annotations and marker styling
  ([`ed575f9`](https://gitlab.com/suturina-group/simpnmr/-/commit/ed575f9d3d2d9516e288ea93dcd9c6016415597c))

- **viz**: Refine fitted shifts header table labels
  ([`951d83a`](https://gitlab.com/suturina-group/simpnmr/-/commit/951d83aecf265cba0a41d1bc04535931f06c99e7))

- **viz**: Refine marker styling in correlation-time plots
  ([`bb8518f`](https://gitlab.com/suturina-group/simpnmr/-/commit/bb8518f682b0462a488b49c45a4ff9dac3799bdc))

- **viz**: Refine susceptibility axis labels, padding, and legend layout
  ([`ff4e04b`](https://gitlab.com/suturina-group/simpnmr/-/commit/ff4e04b91bc1bd8bb33ea28d9a9c6fc4cc8f366a))

- **viz**: Rename fit stats header to fit statistics
  ([`42c62a6`](https://gitlab.com/suturina-group/simpnmr/-/commit/42c62a652053dbe09d8e1322502021ea7ea85c5a))

- **viz**: Rename fit stats header to fit statistics
  ([`50b4dbd`](https://gitlab.com/suturina-group/simpnmr/-/commit/50b4dbd2aa0e9d7e1274dd249c081f83371d20c5))

- **viz**: Rename shift contribution labels to FC and PCS
  ([`ae3e14b`](https://gitlab.com/suturina-group/simpnmr/-/commit/ae3e14bac079fd86c0cbb6f00b28cc091f557705))

- **viz**: Rename susceptibility fit legend entry to Fit
  ([`84fd3a0`](https://gitlab.com/suturina-group/simpnmr/-/commit/84fd3a0f2a945083683bee27b2e9ad5bc8e81363))

- **viz**: Rename susceptibility fit legend entry to Fit
  ([`54ea9b5`](https://gitlab.com/suturina-group/simpnmr/-/commit/54ea9b57181de2d1c52469315431e450a13e6a34))

- **viz**: Replace magnetic susceptibility label with chi symbol
  ([`778ae55`](https://gitlab.com/suturina-group/simpnmr/-/commit/778ae55cbdb1e3209e14ce4420e82eaa29877701))

- **viz**: Simplify fitted shifts summary precision
  ([`99e247f`](https://gitlab.com/suturina-group/simpnmr/-/commit/99e247f7c13a351009b263f9e91ff87f0b761ac3))

- **viz**: Update corr-time contribution plot component colours
  ([`f2c7b01`](https://gitlab.com/suturina-group/simpnmr/-/commit/f2c7b011395dc03b5619926dfea01c2e7798989a))

### Documentation

- **build**: Improve HFC assembly helper docstrings
  ([`3d42853`](https://gitlab.com/suturina-group/simpnmr/-/commit/3d42853845bda901623cfe8c40e97ebc2ade02aa))

- **input-files**: Align susceptibility and paramagnetic centre contracts
  ([`354ddf4`](https://gitlab.com/suturina-group/simpnmr/-/commit/354ddf4f9533d769d82fe8312d69fb5c5a05da30))

- **logging**: Clarify active temperature selection on susceptibility-experiment mismatch
  ([`0ba203d`](https://gitlab.com/suturina-group/simpnmr/-/commit/0ba203df20cc79894940565322e103a2106669cd))

- **tutorials**: Add downloadable examples landing page and usage guide
  ([`80d29b2`](https://gitlab.com/suturina-group/simpnmr/-/commit/80d29b29ad5ac5abee0d8211a2c0616658cb12d4))

- **user-guide**: Document paramagnetic centre input
  ([`0743a12`](https://gitlab.com/suturina-group/simpnmr/-/commit/0743a123d7d59124bdcd950a34592d316de6df90))

### Features

- **app**: Add paramagnetic centre loader
  ([`65d2ab7`](https://gitlab.com/suturina-group/simpnmr/-/commit/65d2ab771bf5b9f1f054cfe9d4f527540ac2d3db))

- **app**: Load ORCA chi-source geometry into Molecule
  ([`826927c`](https://gitlab.com/suturina-group/simpnmr/-/commit/826927cf826bbdb02ebbdaf75b56a18d3a1f7b2b))

- **app**: Skip ab initio g-tensor loading when susceptibility file is absent
  ([`3bc1aef`](https://gitlab.com/suturina-group/simpnmr/-/commit/3bc1aef329b761c2fb2801e08b8cd465a308a5a7))

- **cfg**: Add optional hyperfine spin/orbit/J keywords to FitCorrTimeConfig
  ([`e33a276`](https://gitlab.com/suturina-group/simpnmr/-/commit/e33a27623fe51189462df6522ded2353b6ba6ea8))

- **cfg**: Add paramagnetic centre support to fit_susc config
  ([`001290b`](https://gitlab.com/suturina-group/simpnmr/-/commit/001290bfe1b99c3595f130c7a56a5e1aec33471a))

- **cfg**: Make prediction susceptibility file optional
  ([`d503b2a`](https://gitlab.com/suturina-group/simpnmr/-/commit/d503b2aadd6c1ac9da6bf222c470d4e7fa372abd))

- **cfg**: Rename relaxation electron_coords to hyperfine paramagnetic_centre
  ([`17e22f9`](https://gitlab.com/suturina-group/simpnmr/-/commit/17e22f9a36c06ce21bea59fe22236b1c1031a682))

- **core**: Add shared relaxation formalism evaluator
  ([`1a50280`](https://gitlab.com/suturina-group/simpnmr/-/commit/1a50280983ddc4ff55fe33e8b7805111ad241b21))

- **core**: Add spin-only isotropic susceptibility builder
  ([`6e7eac8`](https://gitlab.com/suturina-group/simpnmr/-/commit/6e7eac8d9ede7eb9fe60e26e74fb47c142aee576))

- **csv**: Export canonical HFC payload for all available labels
  ([`3555473`](https://gitlab.com/suturina-group/simpnmr/-/commit/35554733c13d99286b28ac0d3bee37d6403d7d2e))

- **csv**: Export full geometry and canonical HFC payload
  ([`b38876b`](https://gitlab.com/suturina-group/simpnmr/-/commit/b38876b6dc5a4ceb1f64967d014f922134e6b2a9))

- **csv**: Export full molecular geometry in molecule CSV
  ([`19edc85`](https://gitlab.com/suturina-group/simpnmr/-/commit/19edc85f919a059ce8b217de28725e5f8e5dc722))

- **csv**: Export orbital hyperfine tensor components when available
  ([`1771ca2`](https://gitlab.com/suturina-group/simpnmr/-/commit/1771ca291d1f8a94774a9a779b5e4a0910859400))

- **domain**: Add canonical available HFC store to Molecule
  ([`9a7a27c`](https://gitlab.com/suturina-group/simpnmr/-/commit/9a7a27c1427e7a975788ad2b7f755598ed076d58))

- **domain**: Add canonical HFC store projection to Molecule
  ([`2b96061`](https://gitlab.com/suturina-group/simpnmr/-/commit/2b960615b16e157fbd89f853ccb7c8674681973b))

- **domain**: Add chi_source_coords to Molecule
  ([`ad9e1a2`](https://gitlab.com/suturina-group/simpnmr/-/commit/ad9e1a2fa065366bd39dc1b031c7698ad72ee478))

- **domain**: Add chi_source_labels to Molecule
  ([`4298c10`](https://gitlab.com/suturina-group/simpnmr/-/commit/4298c10dbea8af8a90457e59a3d3033ad644aed6))

- **domain**: Add optional relaxation state to Molecule
  ([`8a7066a`](https://gitlab.com/suturina-group/simpnmr/-/commit/8a7066a0524e8e66bc8ecbd667a2ccdfc50d1a41))

- **domain**: Add paramagnetic centre container to molecule
  ([`e137c2f`](https://gitlab.com/suturina-group/simpnmr/-/commit/e137c2f64ec1e97e55d8889e89418696edb1083c))

- **domain**: Support canonical chi-frame geometry and HFC state in Molecule
  ([`3eed73c`](https://gitlab.com/suturina-group/simpnmr/-/commit/3eed73c1d8441bc65e00b07e47edc2a26fecb99c))

- **fit**: Add math chemical label support to correlation-time diagnostics plots
  ([`6a40328`](https://gitlab.com/suturina-group/simpnmr/-/commit/6a4032819450bfb6fa68e36bd0541391accb518f))

- **io**: Export canonical FC provenance and g-correction diagnostics to molecule CSV
  ([`c6c2577`](https://gitlab.com/suturina-group/simpnmr/-/commit/c6c2577dfc6851d0bd153a6626a1f1e00f7b9e9a))

- **io,fit**: Export spin metadata with chiT regression CSV output
  ([`673a4ce`](https://gitlab.com/suturina-group/simpnmr/-/commit/673a4ceb9493727da157663f550cc589d9159c89))

- **io/csv**: Allow structure-only molecule CSV files
  ([`50aad38`](https://gitlab.com/suturina-group/simpnmr/-/commit/50aad382d45b4bc2e85520de0cb5f26af4bfa798))

- **viz**: Add compact uncertainty formatter and integrate it into fitted shifts
  ([`3f968fd`](https://gitlab.com/suturina-group/simpnmr/-/commit/3f968fd5ee955a0d71e33900eab0a2d12542c8c6))

- **viz**: Add obstacle-aware _place_labels helper for fitted shifts
  ([`68199bb`](https://gitlab.com/suturina-group/simpnmr/-/commit/68199bb27a708dbc4acac32d356f7a79ceb36a45))

- **viz**: Add orbital shift distance-dependence plotting module
  ([`0ca452b`](https://gitlab.com/suturina-group/simpnmr/-/commit/0ca452bca9950e854b8dc6ce6578258fe2e27df4))

- **viz**: Add shared compact table renderer for reusable plot summary layouts
  ([`dd4cbc7`](https://gitlab.com/suturina-group/simpnmr/-/commit/dd4cbc78c4b69539851e5af93edb929326798a58))

- **viz**: Add show_point_labels toggle and marker legend for fitted shifts
  ([`373798c`](https://gitlab.com/suturina-group/simpnmr/-/commit/373798c2c67abfb022cabc5d714a3d9f66065809))

- **viz**: Add soft grid styling to fitted shifts plot
  ([`a7d4c16`](https://gitlab.com/suturina-group/simpnmr/-/commit/a7d4c16692b4930a2b5ab0659e3036fe7ba3fa55))

- **viz**: Add vertical_extended figure size variant
  ([`93a934a`](https://gitlab.com/suturina-group/simpnmr/-/commit/93a934afd7f640f7e5d15e374d626ecf7168e8dc))

- **viz**: Plot orbital shift distance dependence when orbital contribution is available
  ([`b9127cf`](https://gitlab.com/suturina-group/simpnmr/-/commit/b9127cf8245a466e2ba365e42d89ddedc3f48ac6))

- **viz**: Scale interactive preview and default pdf bbox to None
  ([`cefcdfb`](https://gitlab.com/suturina-group/simpnmr/-/commit/cefcdfb03956c367c55e7b3add4c8297e26cae2c))

### Refactoring

- **app**: Bind runtime plot visibility and project output directory in tau-fit pipeline
  ([`76e3079`](https://gitlab.com/suturina-group/simpnmr/-/commit/76e3079ad18af28182c4ddf72bd0b2c4ca180cb0))

- **app**: Clarify susceptibility loader orchestration
  ([`39332ca`](https://gitlab.com/suturina-group/simpnmr/-/commit/39332ca9fe20d2a9e8d1ce80bcaa63301a0517a1))

- **app**: Load paramagnetic centre through dedicated loader in correlation-time fit
  ([`0403af6`](https://gitlab.com/suturina-group/simpnmr/-/commit/0403af633cc4f2862de4f5d81b72f79f973f28e9))

- **app**: Load paramagnetic centre through dedicated loader in predict
  ([`7e7e8af`](https://gitlab.com/suturina-group/simpnmr/-/commit/7e7e8af5eeb3e9c306dbae456890afb1baaf6b65))

- **app**: Load paramagnetic centre through dedicated loader in susceptibility fit
  ([`93526cf`](https://gitlab.com/suturina-group/simpnmr/-/commit/93526cf24ff092bd87ba0082e4e5eb2f2345b554))

- **app**: Make hyperfine loader use molecule paramagnetic centre
  ([`3e7344b`](https://gitlab.com/suturina-group/simpnmr/-/commit/3e7344b812a2d143c73305e0128dd144815f6cea))

- **app**: Move chi-frame preparation toward domain-based flow
  ([`4476b25`](https://gitlab.com/suturina-group/simpnmr/-/commit/4476b25a2cf88aeffbf001371cdb0de40d5a1d4b))

- **app**: Remove obsolete susceptibility iso mode handling from loader
  ([`e9eccd8`](https://gitlab.com/suturina-group/simpnmr/-/commit/e9eccd87528c63974b9abd0f8ce9dd4ec384b4f5))

- **app**: Remove obsolete susceptibility iso mode policy
  ([`b3c68fb`](https://gitlab.com/suturina-group/simpnmr/-/commit/b3c68fb590c4a3873f8937aa118d586c30ee3972))

- **app**: Route csv susceptibility loading through builder
  ([`a45ca8c`](https://gitlab.com/suturina-group/simpnmr/-/commit/a45ca8c077c53ee612bdd9d43302320cedf25ef8))

- **app**: Unify paramagnetic centre as single source of truth
  ([`89a0216`](https://gitlab.com/suturina-group/simpnmr/-/commit/89a02168edd045e3df9921e16136b8c9a525f5fb))

- **app**: Use shared relaxation evaluator in prediction pipeline
  ([`3c7e9f2`](https://gitlab.com/suturina-group/simpnmr/-/commit/3c7e9f2e2bf99ad55652fcedb5a29e18a87ccc8b))

- **build**: Project runtime HFC from canonical molecule store
  ([`674b6ce`](https://gitlab.com/suturina-group/simpnmr/-/commit/674b6ce462c53a0efc5182e71b3d0042b161aa28))

- **cfg,core,app**: Unify paramagnetic centre as single source of truth
  ([`1685344`](https://gitlab.com/suturina-group/simpnmr/-/commit/1685344f73f19d57bf3d721e2c1f3b578d331041))

- **cfg,docs**: Drop relaxation magnetic_field_tesla from PredictConfig and YAML schema
  ([`d7e5468`](https://gitlab.com/suturina-group/simpnmr/-/commit/d7e54681add4f44690bc614f9412c2e3d83d7026))

- **cli**: Switch tau_fit to corr_time_fit module
  ([`39daffb`](https://gitlab.com/suturina-group/simpnmr/-/commit/39daffb0e77845951a7668342fd2b9cca40a3ec0))

- **coords**: Remove config and IO from chi-frame transform helpers
  ([`76da6d5`](https://gitlab.com/suturina-group/simpnmr/-/commit/76da6d5b2b6be97cb28ad8916912305aee903ce6))

- **core**: Add canonical susceptibility iso and FC g-correction diagnostics to domain
  ([`a2967c2`](https://gitlab.com/suturina-group/simpnmr/-/commit/a2967c21ecb1b2da81c22a8d8ab9a2d1821d1a3c))

- **core**: Make point-dipole builder use molecule paramagnetic centre
  ([`6d5c9d3`](https://gitlab.com/suturina-group/simpnmr/-/commit/6d5c9d3944bba037af324334a9690adeb462f262))

- **core**: Make susceptibility builder assign canonical, spin-only, and g-corrected iso
  ([`e3ae5fb`](https://gitlab.com/suturina-group/simpnmr/-/commit/e3ae5fb0a96e1ced9960da8b48ab0bac5c55e9a5))

- **core**: Move susceptibility physics helpers to phys layer
  ([`a430b67`](https://gitlab.com/suturina-group/simpnmr/-/commit/a430b67ccaea202dbd776642dda249594ecbf4a5))

- **core**: Move transform helpers to core and remove config coupling
  ([`97d3620`](https://gitlab.com/suturina-group/simpnmr/-/commit/97d3620a4ad3194e792dabe7bc8db6b9f4b9cb4a))

- **corr-time**: Unify R1 fit-mode execution in shared helper
  ([`15c4498`](https://gitlab.com/suturina-group/simpnmr/-/commit/15c449801c291de806d010fd662a43d63b8a0d61))

- **corr-time**: Wire fitted R1 decomposition into contribution plot
  ([`1dfaf17`](https://gitlab.com/suturina-group/simpnmr/-/commit/1dfaf17c78c71b0d68c6f2d73f661e3d317c65bd))

- **csv**: Simplify corr-time fit diagnostics export
  ([`28e6fbe`](https://gitlab.com/suturina-group/simpnmr/-/commit/28e6fbe3132626298d90418d672dc5fe1f3afed2))

- **domain**: Add placeholder state module for electronic and spin-Hamiltonian containers
  ([`79ed2cd`](https://gitlab.com/suturina-group/simpnmr/-/commit/79ed2cdac67ca5d491ab282db8f78aa40b149483))

- **domain**: Remove relaxation placeholder from molecule module
  ([`05d81a7`](https://gitlab.com/suturina-group/simpnmr/-/commit/05d81a76e7fbdb874d7700e57f9bdeb3f39494a9))

- **examples**: Remove outdated examples and align remaining workflows with current contracts
  ([`cb10ac8`](https://gitlab.com/suturina-group/simpnmr/-/commit/cb10ac802b6e436a732094619158e88bf632853a))

- **fit**: Move correlation-time plotting into viz helpers
  ([`df5c876`](https://gitlab.com/suturina-group/simpnmr/-/commit/df5c8761aee63aebc49f7b309f8e9129afc933fc))

- **fit**: Move correlation-time plotting to viz and source HFC from domain
  ([`5266829`](https://gitlab.com/suturina-group/simpnmr/-/commit/52668299886fa09fcadec0a0cd2b10e834d3c4b0))

- **fit**: Move xyz exports to the end of correlation-time pipeline
  ([`5fb2e93`](https://gitlab.com/suturina-group/simpnmr/-/commit/5fb2e9374bf538992f4084d6ff739765e903ac85))

- **fit**: Remove deprecated tau_fit pipeline module
  ([`c2e746e`](https://gitlab.com/suturina-group/simpnmr/-/commit/c2e746ebcef54619c62c2ffa1d183f8c9b7c85c7))

- **fit**: Rename tau_fit pipeline module to corr_time_fit
  ([`ca5ab98`](https://gitlab.com/suturina-group/simpnmr/-/commit/ca5ab98b64f7f0b357efc6d41b8e0fc261dc5b8a))

- **fit**: Update corr_time CSV writer import to new module location
  ([`730053f`](https://gitlab.com/suturina-group/simpnmr/-/commit/730053fe4a09ce76f4fe8f8b0dd8bdff22ca9e08))

- **fit**: Use shared relaxation evaluator in correlation time fitting
  ([`0b29068`](https://gitlab.com/suturina-group/simpnmr/-/commit/0b2906884cf2497ac5e3653787e5e20cc1f82510))

- **io**: Migrate peak CSV export to molecule-driven averaged shift and linewidth output
  ([`380d739`](https://gitlab.com/suturina-group/simpnmr/-/commit/380d7399596b22c083e4611642eda8eb394cb741))

- **io**: Move correlation-time CSV writer into new corr_time module
  ([`5323033`](https://gitlab.com/suturina-group/simpnmr/-/commit/53230333c54f68eaa281d9b442e1c2d9ee5fcf57))

- **io**: Remove legacy paths from molecule csv reader
  ([`dcb213d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dcb213d51745dd9296bcfcc765a0bb4e8a94987c))

- **io**: Remove legacy relax CSV module
  ([`cb8a231`](https://gitlab.com/suturina-group/simpnmr/-/commit/cb8a231ac9e7a76e7bac3bb8e5599c5227d18e77))

- **io**: Remove unused delimiter from correlation time CSV export
  ([`742dc7e`](https://gitlab.com/suturina-group/simpnmr/-/commit/742dc7ea7679a0ec3574b1eba3bb950ce08099f7))

- **io**: Rename Gaussian gateway raw hyperfine variables to fc/sd
  ([`b812091`](https://gitlab.com/suturina-group/simpnmr/-/commit/b8120910da7f8d8a5fe7c029f6a46db8d885172b))

- **io**: Rename Gaussian hyperfine parser outputs to canonical component names
  ([`aab0e20`](https://gitlab.com/suturina-group/simpnmr/-/commit/aab0e204c8960b5293f0fb9ad895f382f6eea1aa))

- **predict**: Centralize linewidth policy and move peak CSV export to run_predict
  ([`bc27eaa`](https://gitlab.com/suturina-group/simpnmr/-/commit/bc27eaae2e247e137e629c6af2bcf97a1763fa9f))

- **predict**: Replace duplicate relaxation config.magnetic_field_tesla with canonical
  experiment.magnetic_field
  ([`eb76c3b`](https://gitlab.com/suturina-group/simpnmr/-/commit/eb76c3b50bd0cdd91d0cff1df53f923e214d38c7))

- **predict**: Replace resolve_susceptibilities with load_susceptibilities
  ([`0e98c2d`](https://gitlab.com/suturina-group/simpnmr/-/commit/0e98c2d3f66d01845b6dd112bb307defda46afb3))

- **predict,cfg,docs**: Replace relaxation temperature config with canonical experiment.temperature
  ([`5101a93`](https://gitlab.com/suturina-group/simpnmr/-/commit/5101a93e7f50bddf9219f44b5e293dba9012fe0b))

- **relaxation**: Preserve decomposition channels across evaluation and fit workflows
  ([`6a3fbc5`](https://gitlab.com/suturina-group/simpnmr/-/commit/6a3fbc578b4120e67c70dbc5fe6986a42e3cd4e7))

- **susc**: Restore fc shift contribution output wiring
  ([`309e906`](https://gitlab.com/suturina-group/simpnmr/-/commit/309e9063da416695b04f486f56d8b35c7770a4db))

- **susc**: Separate chi tensor/iso builders and honor csv chi_iso loading
  ([`8720fa9`](https://gitlab.com/suturina-group/simpnmr/-/commit/8720fa9bed997e116e5406ffc6c8cb5682092cb3))

- **viz**: Adapt correlation-time plots to paper-style layout
  ([`89b3b9c`](https://gitlab.com/suturina-group/simpnmr/-/commit/89b3b9c070f89b1dccf2a227d9097ca140c175f0))

- **viz**: Add R1 contribution plot for corr-time diagnostics
  ([`e66e75f`](https://gitlab.com/suturina-group/simpnmr/-/commit/e66e75fc286a4436d8818c618c5d9d525df44b2d))

- **viz**: Extract fitted shifts plot into dedicated module
  ([`ce3c198`](https://gitlab.com/suturina-group/simpnmr/-/commit/ce3c198ac75137388140af4718ac2db94afb2e1c))

- **viz**: Extract fitted shifts plot into dedicated module
  ([`069d41f`](https://gitlab.com/suturina-group/simpnmr/-/commit/069d41fad5df25bb4ef86cc606de818ac7de3f3b))

- **viz**: Improve fitted-shift plot layout and annotation styling
  ([`f943843`](https://gitlab.com/suturina-group/simpnmr/-/commit/f94384393535ed1a90f876c0e59d5e359ea535c7))

- **viz**: Integrate shared compact table renderer into corr time plots
  ([`61b74b0`](https://gitlab.com/suturina-group/simpnmr/-/commit/61b74b030138f2ed62d73d46bd8fdcb2c7f8c132))

- **viz**: Integrate shared compact table renderer into fitted shifts plot
  ([`c109e5b`](https://gitlab.com/suturina-group/simpnmr/-/commit/c109e5b4a0e3c8ad043a60af3ab74e2ec7e3fc01))

- **viz**: Integrate shared compact table renderer into susceptibility plots
  ([`fa45ce1`](https://gitlab.com/suturina-group/simpnmr/-/commit/fa45ce1d036cf1578be8427e7a6d10583ebc8458))

- **viz**: Introduce canonical canvas helpers and figure size registry
  ([`0aef5ba`](https://gitlab.com/suturina-group/simpnmr/-/commit/0aef5ba989462b4841ab22e698d3949b402b2c8d))

- **viz**: Move scatter label layout from fittes_shifts to utility
  ([`e2ebf46`](https://gitlab.com/suturina-group/simpnmr/-/commit/e2ebf46fb07877b3f9757231d837ea8ac1e8b20b))

- **viz**: Move scatter label layout from fittes_shifts to utility
  ([`9d40881`](https://gitlab.com/suturina-group/simpnmr/-/commit/9d408811c9cd74faac09405f8265fac3a49feff6))

- **viz**: Move shared label layout resolver into layout module
  ([`b405b1f`](https://gitlab.com/suturina-group/simpnmr/-/commit/b405b1fd50e74b6c2345888a09434ad5f8efec1f))

- **viz**: Redesign fitted shifts plot layout and helpers
  ([`f921a61`](https://gitlab.com/suturina-group/simpnmr/-/commit/f921a61f37393d53810e1568b8f722e51b22cc64))

- **viz**: Refine correlation-time diagnostic plot styling and label layout
  ([`aeaa84c`](https://gitlab.com/suturina-group/simpnmr/-/commit/aeaa84c0a4f2b9346e95f8b23449bb86b79fa88f))

- **viz**: Remove deprecated label_layout utility module
  ([`9822e5d`](https://gitlab.com/suturina-group/simpnmr/-/commit/9822e5d419e556c2651515e00a23d643c50da9d7))

- **viz**: Remove highlight peak-position markers from spectrum plots
  ([`e70ce3e`](https://gitlab.com/suturina-group/simpnmr/-/commit/e70ce3e896d656acb3da85aebdf956a4742203ef))

- **viz**: Remove table-specific typography scale
  ([`7d36b87`](https://gitlab.com/suturina-group/simpnmr/-/commit/7d36b87d0ebedd2f790288881d8c686f7562c52c))

- **viz**: Remove tight-layout and subplot-adjust fallbacks
  ([`8de21a6`](https://gitlab.com/suturina-group/simpnmr/-/commit/8de21a672c6ac48d60bcb042bd25e9549541de26))

- **viz**: Split correlation time diagnostics into dedicated plot helpers
  ([`32deeb4`](https://gitlab.com/suturina-group/simpnmr/-/commit/32deeb4571806dc114c0dc0b3fddf09502a41740))

- **viz**: Update fitted shifts marker styling
  ([`fabecb6`](https://gitlab.com/suturina-group/simpnmr/-/commit/fabecb6bba72d239955dabb57c55b5e82f4044b1))

- **viz**: Widen canonical vertical figure size from 8 cm to 9 cm
  ([`eeae670`](https://gitlab.com/suturina-group/simpnmr/-/commit/eeae670cae8868c38ac541865f21bbf4bbcd276f))

- **vt**: Move ab initio chiT helper into vt_fit and document fit_vt
  ([`97a9fad`](https://gitlab.com/suturina-group/simpnmr/-/commit/97a9fad1f9fe09c47532a7295dfff857cd9952b6))

### Testing

- Add library-grade test skeleton for pipelines, sources, and regression
  ([`4979351`](https://gitlab.com/suturina-group/simpnmr/-/commit/4979351c1828675e740e539a82054e884fb820af))

- **integration**: Add canonical fit_corr_time CLI pipeline test
  ([`62d2950`](https://gitlab.com/suturina-group/simpnmr/-/commit/62d2950228ba8502a33787db40bb0b1c6daad40e))

- **integration**: Add canonical happy-path coverage for fit_susc pipeline
  ([`ea74695`](https://gitlab.com/suturina-group/simpnmr/-/commit/ea74695941a896fa59d129eeedbf4ee84ea99c62))

- **integration**: Add canonical happy-path coverage for predict pipeline
  ([`a3ee12e`](https://gitlab.com/suturina-group/simpnmr/-/commit/a3ee12e9fbdb0d074305b2d495824a1065881610))

- **integration**: Add cli coverage for spinham extraction
  ([`927adcb`](https://gitlab.com/suturina-group/simpnmr/-/commit/927adcb9c242f26b13776798d02caf0a7df3229c))

- **integration**: Add csv and nevpt2 susceptibility coverage for pcs isosurface cli
  ([`45f2b0d`](https://gitlab.com/suturina-group/simpnmr/-/commit/45f2b0d1a45120cec9b8da44b2af84f658bbd4bc))

- **pytest**: Configure default pytest output options
  ([`7ab4ad3`](https://gitlab.com/suturina-group/simpnmr/-/commit/7ab4ad363b229f92ff92a35c2ee9ef10be808563))

- **tests**: Update test suite for current contracts (wip)
  ([`490ccc9`](https://gitlab.com/suturina-group/simpnmr/-/commit/490ccc940a8db50e7530060d22bcc09fe7ee34d2))


## v1.6.0 (2026-03-10)

### Documentation

- Clarify orbital contribution support for ORCA 5/6 outputs
  ([`c89eadd`](https://gitlab.com/suturina-group/simpnmr/-/commit/c89eadd8284b4d3e70495028985c8fa4ae0131f2))

### Features

- **csv**: Export total orbital shift contribution
  ([`ad20f88`](https://gitlab.com/suturina-group/simpnmr/-/commit/ad20f88cd2145c1ceaac3b4d53573612a4b1b3ba))

### Refactoring

- **csv**: Define molecule export columns via specs
  ([`a816f06`](https://gitlab.com/suturina-group/simpnmr/-/commit/a816f06eea2f213fb4e65cd5e48ef07d2d8caff5))

### Testing

- Remove overly brittle tests
  ([`50b4301`](https://gitlab.com/suturina-group/simpnmr/-/commit/50b4301ecac227f613598fcebf0925f524d6e08b))


## v1.5.0 (2026-03-09)

### Bug Fixes

- Resolve several bugs preventing Hungarian assignment from working
  ([`f5ab4fa`](https://gitlab.com/suturina-group/simpnmr/-/commit/f5ab4fa1b313cc664133327a1751d5a24b5e4ef0))

- **app**: Fix assignment behavior in susc fitting
  ([`1b0d3df`](https://gitlab.com/suturina-group/simpnmr/-/commit/1b0d3df668e8f607ece6e4bbd00b1d1e8d276788))

- **app**: Restore fixed assignment handling in susceptibility fitting pipeline
  ([`84cdb29`](https://gitlab.com/suturina-group/simpnmr/-/commit/84cdb296e7d79c297c853805dfc275fcbc45b22b))

- **core**: Prevent domain mutation during Hungarian search and restore best fit state instead of
  leaking the last fit
  ([`6f6a2fe`](https://gitlab.com/suturina-group/simpnmr/-/commit/6f6a2feaa1b96349f2f336a61480e6fd23b1ce6e))

- **io**: Drop empty rows when loading experiment CSV files
  ([`aaed38d`](https://gitlab.com/suturina-group/simpnmr/-/commit/aaed38d6a077e59068ba5a2d32b51519c09e92a6))

### Chores

- Remove stray file
  ([`e506c54`](https://gitlab.com/suturina-group/simpnmr/-/commit/e506c548a55bc410e6c5b641b22b7c86fcc3181e))

- **governance**: Introduce AI contribution contract and MR template
  ([`f31fb82`](https://gitlab.com/suturina-group/simpnmr/-/commit/f31fb827faf035e23b4444c3a253f29428074acd))

### Documentation

- **user-guide**: Document Hungarian assignment search contract and defaults
  ([`174dd38`](https://gitlab.com/suturina-group/simpnmr/-/commit/174dd380d8eaeddcfb3556376ffe1d96d25df783))

- **user-guide**: Fix internal warning formatting in theory docs
  ([`5e35e40`](https://gitlab.com/suturina-group/simpnmr/-/commit/5e35e40aaefddde76cf3ebb51384ed2dbddce69b))

### Features

- **app**: Add assignment search policy presets and resolver
  ([`f9cc66d`](https://gitlab.com/suturina-group/simpnmr/-/commit/f9cc66d5667fb0ef605eb6e7827c72c7a220ff2a))

- **fitting**: Add Hungarian assignment method for fit_susc
  ([`60f933e`](https://gitlab.com/suturina-group/simpnmr/-/commit/60f933e6206de4266b30235cd57db97e607b6f86))

### Refactoring

- **cfg**: Enforce Hungarian search mapping contract in fit config
  ([`ae467c1`](https://gitlab.com/suturina-group/simpnmr/-/commit/ae467c16a039aa75500b6f592c4911530805b1ae))

- **core**: Move assignment algorithms from app pipeline to core fitting
  ([`a116cb8`](https://gitlab.com/suturina-group/simpnmr/-/commit/a116cb861982c24b6420839b419f4230b177af86))

### Testing

- **integration**: Disable Hungarian vs permute equivalence test pending fixture migration
  ([`7a9de1c`](https://gitlab.com/suturina-group/simpnmr/-/commit/7a9de1cb8fe1cd43b37e8fce6b5ee6d887487217))

- **integration**: Fix Hungarian test setup and temporarily disable brittle test pending refactor
  ([`b7ce49c`](https://gitlab.com/suturina-group/simpnmr/-/commit/b7ce49c60cd65f9814e96857d37259e2284dc601))


## v1.4.0 (2026-02-20)

### Documentation

- **architecture**: Document application-level policy layer
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **user**: Add standalone CLI utilities to user guide index
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **user**: Mark susceptibility format as optional with automatic method selection
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

### Features

- **policies**: Add susceptibility policy for backend and ORCA method resolution
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

### Refactoring

- **config**: Make susceptibility format optional and legacy-compatible
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **csv**: Avoid absolute paths in exported CSV comments
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **loaders**: Delegate susceptibility backend and ORCA section resolution to policy
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **pcs_iso**: Make susceptibility method autodetected
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **predict**: Use susceptibility policy for ORCA routing
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **susceptibility**: Autodetect backend and ORCA method via policy layer
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))


## v1.3.5 (2026-02-20)

### Bug Fixes

- **io**: Remove hardcoded linewidth and use relaxation-derived values when available
  ([`96b27da`](https://gitlab.com/suturina-group/simpnmr/-/commit/96b27da175af7c98e9ab5bba6345c599d861c095))

### Chores

- **app**: Delegate CSV comment prefix handling to write_csv_safe
  ([`509ba9d`](https://gitlab.com/suturina-group/simpnmr/-/commit/509ba9d10dc18ae1816452ae5c717935eaafcdc8))

- **app**: Delegate CSV comment prefix handling to write_csv_safe
  ([`f920b11`](https://gitlab.com/suturina-group/simpnmr/-/commit/f920b113a6ade5b25cc9a219051a5c95d2bd7889))

- **csv**: Introduce write_csv_safe for standardized CSV output
  ([`a5c6328`](https://gitlab.com/suturina-group/simpnmr/-/commit/a5c6328a4c4d8a895905c76962993f68a1143e26))

- **io**: Unify CSV exports via write_csv_safe for safe encoding
  ([`5fd6d40`](https://gitlab.com/suturina-group/simpnmr/-/commit/5fd6d404065c6c06708c86460a1d49d9d28aec34))

- **viz**: Add label font size to typography scale
  ([`afa2763`](https://gitlab.com/suturina-group/simpnmr/-/commit/afa276344f0f0ea318452259721ffbc6b8b701bb))

- **viz**: Increase spectrum label sizes and standardize CSV export via write_csv_safe
  ([`45e7472`](https://gitlab.com/suturina-group/simpnmr/-/commit/45e747232b9e04eeb2ba90b665b265198b246334))

- **viz**: Update legacy PNG references to PDF across the repository
  ([`7ef07a7`](https://gitlab.com/suturina-group/simpnmr/-/commit/7ef07a76c8901070427239a0c15ced06f4760be6))

### Refactoring

- **io**: Move write_spectrum helper into csv spec module
  ([`8dafc78`](https://gitlab.com/suturina-group/simpnmr/-/commit/8dafc7803711f5263a45b429de13ee122c6e946a))

### Testing

- **integration**: Add --hide flag to CLI example tests
  ([`1c537ec`](https://gitlab.com/suturina-group/simpnmr/-/commit/1c537ec97a1a1baed4d8a28abbf7073aa4daf9e3))

- **pytest**: Remove addopts marker filtering
  ([`256d640`](https://gitlab.com/suturina-group/simpnmr/-/commit/256d6400531df4c1685f43d6009bb82f665a9275))

- **unit**: Add encoding tests for CSV read/write helpers
  ([`be92d13`](https://gitlab.com/suturina-group/simpnmr/-/commit/be92d131bd682150abbef05f3e4dd351489d54ed))


## v1.3.4 (2026-02-16)

### Bug Fixes

- Trigger patch release
  ([`9439c7b`](https://gitlab.com/suturina-group/simpnmr/-/commit/9439c7b7cef89035310ec1dd1b0cfcfee76a25aa))


## v1.3.3 (2026-02-16)

### Bug Fixes

- **susc**: Correct VT fitting and uncertainty propagation
  ([`e3ce7d1`](https://gitlab.com/suturina-group/simpnmr/-/commit/e3ce7d170a500ae182deac6a95fda3bd29d00e88))


## v1.3.2 (2026-02-10)

### Bug Fixes

- Trigger patch release
  ([`c76aa10`](https://gitlab.com/suturina-group/simpnmr/-/commit/c76aa108bfd7de61676de4a780c4b9c2ab74494b))


## v1.3.1 (2026-02-05)

### Bug Fixes

- **csv**: Fix delimiter handling for raw experiment CSV
  ([`3d4c129`](https://gitlab.com/suturina-group/simpnmr/-/commit/3d4c129efbec704c70ad6859c2b2ab6e664e9453))

### Chores

- **docs**: Minor clarifications and wording improvements
  ([`c975f57`](https://gitlab.com/suturina-group/simpnmr/-/commit/c975f576a955aa903a1a79fbc800dd16c1d25c6f))

- **packaging**: Add missing __init__.py files to package directories
  ([`d139e0a`](https://gitlab.com/suturina-group/simpnmr/-/commit/d139e0a5867982ca8316201ff43bb15ecf040411))


## v1.3.0 (2026-02-01)

### Documentation

- Switch root_doc back to index
  ([`f86a926`](https://gitlab.com/suturina-group/simpnmr/-/commit/f86a926b9c0acc1c9a4688216ee0e7827376afbd))

### Features

- Architectural refactor of application and IO layers
  ([`5291fea`](https://gitlab.com/suturina-group/simpnmr/-/commit/5291fea3abfcac2951cae9ce14050d4500e869fc))


## v1.2.2 (2025-12-04)

### Bug Fixes

- Enable lanthanide support, Correct metadata handling, Add CSV export for fit_susc_corr_time
  ([`d948a49`](https://gitlab.com/suturina-group/simpnmr/-/commit/d948a4906604162ebc9f410d042a0bcca587f936))


## v1.2.1 (2025-12-04)


## v1.2.0 (2025-12-04)

### Bug Fixes

- Relaxation predictions printed
  ([`e761b87`](https://gitlab.com/suturina-group/simpnmr/-/commit/e761b87b7e2acbb1b3127dc8db46ea3991b0a9af))

### Chores

- Delete reduntand files
  ([`083dc93`](https://gitlab.com/suturina-group/simpnmr/-/commit/083dc9367b6d67f405aa60d8509a37922c1b2410))

### Features

- Correlation time fitting at multiple fields or temperatures
  ([`71daba5`](https://gitlab.com/suturina-group/simpnmr/-/commit/71daba583c8a9b28bff2935916fb258635d83ba2))

- Implement additional relaxation mechanism WIP
  ([`d206671`](https://gitlab.com/suturina-group/simpnmr/-/commit/d2066715873e73b43518a16e67c30fec15d50c30))

- Implement additional relaxation mechanism WIP
  ([`328029c`](https://gitlab.com/suturina-group/simpnmr/-/commit/328029c5183fbf1daa746a559cb3bb2784324626))


## v1.1.1 (2025-11-28)

### Bug Fixes

- **version**: Update SimpNMR version to 1.1.1
  ([`c6bef83`](https://gitlab.com/suturina-group/simpnmr/-/commit/c6bef83ce0886c9d614c494c43a8e710f25bf138))


## v1.1.0 (2025-11-28)

### Bug Fixes

- Update plots in visualise.py (plot_isoaxrho), improve chi_plot.py readability, and fix bug in
  readers.py (read_orca_spin) for reading XYZ files
  ([`9b35e85`](https://gitlab.com/suturina-group/simpnmr/-/commit/9b35e8555bc73785071844d66098490c5d4a5ef8))

### Features

- Add chi-frame geometry plotting and support single-point CSV data in chi_plot.py
  ([`c3580c4`](https://gitlab.com/suturina-group/simpnmr/-/commit/c3580c437e8a08262097715e27dbfee77f55798d))

- **core**: Improve HFC and chi inputs and fix CI pipeline
  ([`4a1d285`](https://gitlab.com/suturina-group/simpnmr/-/commit/4a1d285b31a6c10888b56eb865bfbc4e4f467d4b))


## v1.0.8 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`764dd74`](https://gitlab.com/suturina-group/simpnmr/-/commit/764dd748d3dc1aa8893c0de7bf1edf0f16f542c7))


## v1.0.7 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`057e35c`](https://gitlab.com/suturina-group/simpnmr/-/commit/057e35c1cb7705b3060f09b2074beac5a3e888ca))

- Test semantic-release after fetching tags
  ([`2828a87`](https://gitlab.com/suturina-group/simpnmr/-/commit/2828a87a13316e1b40bb51ad13d5db1729ffd3b2))


## v1.0.6 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`362bceb`](https://gitlab.com/suturina-group/simpnmr/-/commit/362bceb07b4e93b62fbffe97823d8cddf55dd32a))


## v1.0.5 (2025-11-19)


## v1.0.4 (2025-11-19)


## v1.0.3 (2025-11-19)


## v1.0.2 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`921e62c`](https://gitlab.com/suturina-group/simpnmr/-/commit/921e62ca907f6e35aec1e7a4b89269e11bf35320))


## v1.0.1 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`12037ec`](https://gitlab.com/suturina-group/simpnmr/-/commit/12037ec502af7f3754a44fbdca96edab540ffd1c))


## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release
