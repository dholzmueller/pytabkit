# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pytabkit.__about__ import __version__

project = 'pytabkit'
copyright = '2024, David Holzmüller, Léo Grinsztajn, Ingo Steinwart'
author = 'David Holzmüller, Léo Grinsztajn, Ingo Steinwart'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'default'
html_static_path = ['_static']

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
# python_maximum_signature_line_length = 88


# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# following https://stackoverflow.com/questions/10324393/sphinx-build-fail-autodoc-cant-import-find-module
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
