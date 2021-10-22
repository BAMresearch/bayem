import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayes",
    version="0.1",
    author="BAM",
    author_email="thomas.titscher@bam.de",
    description="Variational Bayes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy", "tabulate"],
    extras_require={  # Optional
        "dev": ["black"],
        "test": ["coverage", "pytest", "flake8", "imageio", "matplotlib", "hypothesis"],
        "doc": ["sphinx", "sphinx_rtd_theme", "doit"],
    },
)
