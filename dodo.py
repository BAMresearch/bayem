import glob


def task_website():
    try:
        import sphinx
        import sphinx_rtd_theme
    except ImportError as e:
        raise ImportError(
            "You need to install `sphinx` and "
            "`sphinx_rtd_theme` to build the website locally. Install "
            "them manually or via \n $pip3 install .[doc]"
        ) from e

    pages = glob.glob("docs/*.rst")
    return {
        "file_dep": pages + ["conf.py", "index.rst"],
        "actions": ["sphinx-build . .build"],
        "verbosity": 2,
    }
