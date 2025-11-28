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
        "xyz_py>=5.13.0",
        "pandas",
        "pathos",
        "pyyaml",
        "pyyaml-include",
        "adjustText",
        "extto>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "simpnmr = simpnmr.cli:interface",
            "plot_A_funcs = simpnmr.scripts.batch_hf_plot:main",
            "plot_chi_funcs = simpnmr.scripts.batch_susc_plot:main",
            "chi_plot = simpnmr.scripts.chi_plot:main",
            "get_susc = simpnmr.scripts.get_susc:main",
            "xyz_to_chemlabel = simpnmr.scripts.chemcraft_xyz_to_chemlabels:main",
        ]
    },
)