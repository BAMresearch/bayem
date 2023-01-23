def task_website():
    try:
        import sphinx
        import sphinx_rtd_theme
    except ImportError as e:
        raise ImportError(
            "You need to install `sphinx` and "
            "`sphinx_rtd_theme` to build the website locally. Install "
            "them manually or via \n $pip3 install .[docs]"
        ) from e

    return {
        "file_dep": ["conf.py", "index.rst"],
        "actions": ["sphinx-build . .build"],
        "targets": [".\.build\index.html"],
        "verbosity": 2,
    }
