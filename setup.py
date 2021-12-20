import os
from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname("__file__"))

version = {}
with open(os.path.join(here, "seca", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="SECA",
    version=version["__version__"],
    description="",
    long_description=readme,
    long_description_content_type="text/md",
    url="https://github.com/delftcrowd/SECA",
    author="",
    author_email="",
    license="Apache Software License 2.0",
    packages=find_packages(exclude=["*tests*"]),
    package_data={"seca": ["data/*.csv"]},
    key_words=["machine learning", "explainability",],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    test_suite="tests",
    install_requires=[
        "graphviz",
        "keras",
        "matplotlib",
        "mlxtend",
        "notebook",
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "tensorflow",
        "saliency",
        "symspellpy",
    ],
    extras_require={
        "dev": [
            "black",
            "bump2version",
            "mock",
            "pytest",
            "pytest-cov",
            "jupyter-book>=0.7.0",
            "sphinx-click",
            "sphinx-tabs",
            "sphinxext-rediraffe",
            "sphinx_inline_tabs",
            "ghp-import",
        ]
    },
)
