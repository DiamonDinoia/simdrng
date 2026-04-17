"""Sphinx configuration for simdrng."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# -- Project information -----------------------------------------------------
project = "simdrng"
author = "Marco Barbone and contributors"
copyright = "2026, Marco Barbone and contributors"  # noqa: A001
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "breathe",
    "exhale",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

# -- Breathe / Exhale --------------------------------------------------------
# On RTD we run doxygen ourselves as a pre-build step; locally the CMake
# `docs` target wires BREATHE_PROJECT_XML to the Doxygen output directory.
_here = Path(__file__).resolve().parent
_default_xml = _here.parent / "build" / "docs" / "doxygen" / "xml"
_breathe_xml = os.environ.get("BREATHE_PROJECT_XML", str(_default_xml))

breathe_projects = {"simdrng": _breathe_xml}
breathe_default_project = "simdrng"

exhale_args = {
    "containmentFolder": "./api_cpp_auto",
    "rootFileName": "library_root.rst",
    "doxygenStripFromPath": "..",
    "rootFileTitle": "C++ API Reference",
    "createTreeView": True,
    "exhaleExecutesDoxygen": False,
}

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

primary_domain = "cpp"
highlight_language = "cpp"


def _run_doxygen_if_needed() -> None:
    """On RTD, run Doxygen here so Breathe has XML available."""
    if not os.environ.get("READTHEDOCS"):
        return
    xml_dir = Path(_breathe_xml)
    if xml_dir.exists():
        return
    doxy = _here / "Doxyfile.rtd"
    # Generate a minimal Doxyfile substituting a known input dir.
    xml_dir.parent.mkdir(parents=True, exist_ok=True)
    doxy.write_text(
        "PROJECT_NAME=simdrng\n"
        f"INPUT={_here.parent / 'include'}\n"
        "RECURSIVE=YES\n"
        "FILE_PATTERNS=*.hpp *.h\n"
        "GENERATE_HTML=NO\n"
        "GENERATE_LATEX=NO\n"
        "GENERATE_XML=YES\n"
        f"OUTPUT_DIRECTORY={xml_dir.parent}\n"
        "XML_OUTPUT=xml\n"
        "QUIET=YES\n"
        "EXTRACT_ALL=YES\n"
        "ENABLE_PREPROCESSING=YES\n"
        "MACRO_EXPANSION=YES\n"
    )
    subprocess.run(["doxygen", str(doxy)], check=True)


_run_doxygen_if_needed()
