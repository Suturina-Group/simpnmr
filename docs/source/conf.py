"""Sphinx configuration file for the SimpNMR documentation.

This file defines documentation structure, extensions, theme configuration,
and output formatting for HTML and LaTeX builds.

For the full list of available options, see:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------
# Extend sys.path so Sphinx can import the simpnmr package and its submodules.
import os
import sys

from simpnmr.__version__ import __version__

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../"))

# -- Project metadata -------------------------------------------------------
project = "simpnmr"
copyright = ""

# Title displayed in the HTML documentation header and browser tab.
html_title = f"SimpNMR v{__version__}"
# -- General configuration ---------------------------------------------------

# Sphinx extensions enabled for this documentation build.
# Only extensions that are actively used should be listed here.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinxemoji.sphinxemoji",
    "sphinx_design",
]
# Bibliography configuration (used for citations in the documentation).
bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "super"

napoleon_numpy_docstring = True

# Custom Jinja2 templates for HTML output.
templates_path = ["_templates"]

# Files and directories to ignore during the documentation build.
exclude_patterns = []


# -- HTML output configuration ---------------------------------------------

# PyData Sphinx Theme is used for a clean, modern documentation layout.
html_theme = "pydata_sphinx_theme"

# Use "directory-style" URLs
html_use_directory_uris = True

# Canonical base URL for sitemap/canonical links.
html_baseurl = "https://simpnmr.org/"

# Remove footer metadata (copyright and Sphinx attribution)
html_show_copyright = False
html_show_sphinx = False

# Theme-specific options controlling navigation behaviour and header links.
html_theme_options = {
    "navigation_depth": 2,
    "show_nav_level": 1,
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "text": "SimpNMR",
    },
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/suturina-group/simpnmr",
            "icon": "fa-brands fa-gitlab",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/simpnmr/",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_logo = "_static/simpnmr-logo.png"
html_favicon = "_static/simpnmr-favicon.ico"

# Paths containing custom static assets (CSS, images, JavaScript).
html_static_path = ["_static"]

# Additional CSS files applied after the theme's default styles.
html_css_files = ["style.css"]

# Default options for autodoc-generated API documentation.
autodoc_default_options = {
    "member-order": "bysource",
}

# -- LaTeX / PDF output configuration --------------------------------------

# Settings below control experimental PDF builds via LaTeX.
latex_engine = "xelatex"

latex_elements = {
    "fontpkg": r"""
\setmainfont{Arial}
""",
    "preamble": r"""
\usepackage[titles]{tocloft}
\usepackage{amsmath}
\usepackage{amssymb}
\newcommand{\gt}{>}
\newcommand{\lt}{<}
\renewcommand\AA{\text{Å}}
""",
    "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
    "printindex": r"\footnotesize\raggedright\printindex",
}
latex_show_urls = "footnote"
root_doc = "index"
latex_documents = [
    (root_doc, "simpnmr.tex", "simpnmr Documentation", "", "manual"),
]
