# SimpNMR

<img src="https://gitlab.com/suturina-group/simpnmr/-/raw/main/docs/source/_static/simpnmg-full-logo.png" width="300">

[![Docs](https://img.shields.io/badge/docs-simpnmr.org-blue)](https://simpnmr.org/)
[![PyPI](https://img.shields.io/pypi/v/simpnmr.svg)](https://pypi.org/project/simpnmr/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](https://gitlab.com/suturina-group/simpnmr/-/blob/main/LICENSE)

**SimpNMR** is an open-source Python toolkit for prediction, fitting, and analysis
of paramagnetic NMR spectra based on experimental data and ab initio calculations.

## Features

- Prediction of paramagnetic NMR shifts and spectra
- Susceptibility tensor fitting from experimental measurements
- Integration with quantum chemistry outputs (ORCA, Gaussian, Molcas)

## Installation

```bash
pip install simpnmr
simpnmr --help
```

## Quick example

```bash
simpnmr predict input.yml
```

## Documentation

👉 https://simpnmr.org

## Project status

Active development. Public APIs and configuration schemas are considered stable;
breaking changes are coordinated with maintainers and documented in the changelog.

## Support

Please use GitLab Issues for bug reports and feature requests.

## Contributing

Development guidelines, architecture, and release policies are documented in the
Developer Guide:

👉 https://simpnmr.org/developer_guide/

## Citation

If you use SimpNMR in academic work, please cite the associated preprint and
the specific version used:

> SimpNMR, ChemRxiv (2026). https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001463/v1

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
