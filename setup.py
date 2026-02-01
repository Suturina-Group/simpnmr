"""Package installation and distribution configuration."""

import setuptools

# Read version from simpnmr/__version__.py
version = {}
with open("simpnmr/__version__.py", "r", encoding="utf-8") as f:
    exec(f.read(), version)

setuptools.setup(
    name="simpnmr",
    version=version["__version__"],
    author="Suturina Group",
    author_email="",
    description="A package for working with paramagnetic NMR spectra",
    url="https://gitlab.com/suturina-group/simpnmr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "pandas",
        "pathos",
        "pyyaml",
        "pyyaml-include",
        "adjustText",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "simpnmr = simpnmr.cli.cli:interface",
            "plot_A_funcs = simpnmr.tools.batch_hf_plot:main",
            "plot_chi_funcs = simpnmr.tools.batch_susc_plot:main",
            "xyz_to_chemlabel = simpnmr.tools.coords_tools.chemcraft_xyz:main",
        ]
    },
)
