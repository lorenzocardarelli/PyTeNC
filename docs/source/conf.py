# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))


# -- Project information -----------------------------------------------------

project = 'PyTeNC'
copyright = '2020, Lorenzo Cardarelli'
author = 'Lorenzo Cardarelli'

# The short X.Y version
version = '1.0'
# The full version, including alpha/beta/rc tags
release = '2020'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
]

# todo: When you use the syntax ".. todo:: some todo comment" in your docstring,
# it will appear in a highlighted box in the documentation.
# In order for this extension to work, make sure you include the following:
# If true, 'todo' and 'todoList' produce output, else they produce nothing.
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'
# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# autodoc: allows Sphinx to automatically generate documentation for the docstrings
#          in your code. Get more info here: http://www.sphinx-doc.org/en/master/ext/autodoc.html
# Some useful configurations:
autoclass_content = "both"  # Include both the class's and the init's docstrings.
autodoc_member_order = 'bysource'  # In the documentation, keep the same order of members as in the code.
autodoc_default_flags = ['members']  # Default: include the docstrings of all the class/module members.

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Defining the static path allows me to add my own logo for the project:
#   (make sure the theme of your choice support the use of logo.
#CHANGEME# Add a photo of your choice under _static folder, and link to its name here.
html_logo = 'logo.png'
html_logo = 'PyTeNC_logo.pdf'
html_logo = 'PyTeNC_logo.png'

# Custom sidebar templates, must be a dictionary that maps document names to template names.
# In This project I chose to include in the sidebar:
#   - Table of Contents: I chose globaltoc as it is less refined,
#     and configured the title by editing the globaltoc template (see explanation above, in the templates_path comment)
#   - Search box: appears below the TOc, and can be configured by editing css attributes.
html_sidebars = {
    '**': [
        'globaltoc.html',
        'searchbox.html'
    ]
}


