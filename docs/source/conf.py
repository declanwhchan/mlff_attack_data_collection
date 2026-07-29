# Sphinx configuration for the MLFF robustness research website.

project = "Adversarial Attacks Against Machine Learning Force Fields"
copyright = "2026, Declan Chan"
author = "Declan Chan"
release = "0.1"

root_doc = "index"

extensions = []

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "pydata_sphinx_theme"

html_title = (
    "Adversarial Attacks on Machine Learning Force Fields"
)

html_short_title = "MLFF Robustness"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

html_js_files = [
    "custom.js",
]

html_show_sourcelink = True
html_show_sphinx = False

html_theme_options = {
    "logo": {
        "text": "MLFF Robustness",
    },
    "navbar_align": "left",
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "header_links_before_dropdown": 0,
    "navbar_start": [],
    "navbar_center": [],
    "navbar_persistent": [],
    "navbar_end": [],
    "collapse_navigation": False,
    "navigation_depth": 3,
    "icon_links_label": "Project links",
    "icon_links": [
        {
            "name": "GitHub repository",
            "url": (
                "https://github.com/declanwhchan/"
                "mlff_attack_data_collection"
            ),
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
            "attributes": {
                "target": "_blank",
                "rel": "noopener",
            },
        },
    ],
}


# Display the complete research navigation on the left.
html_sidebars = {
    "**": [
        "search-field.html",
        "sidebar-utilities.html",
        "sidebar-navigation.html",
    ],
}

# reference-target-highlighting
html_static_path = list(globals().get("html_static_path", []))
if "_static" not in html_static_path:
    html_static_path.append("_static")

html_css_files = list(globals().get("html_css_files", []))
if "custom.css" not in html_css_files:
    html_css_files.append("custom.css")
