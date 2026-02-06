"""Sphinx configuration for flashscenic documentation."""

project = "flashscenic"
copyright = "2025, Hao Zhu, Donna Slonim"
author = "Hao Zhu, Donna Slonim"
version = "0.0.2"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# autodoc2 settings
autodoc2_packages = [
    "../flashscenic",
]
autodoc2_render_plugin = "myst"

# Theme
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/haozhu233/flashscenic",
    "use_repository_button": True,
    "use_issues_button": True,
}
html_title = "flashscenic"

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
