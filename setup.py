import os
import sys
from setuptools import setup, find_packages


PACKAGE_NAME = "imvl"
DESCRIPTION = "A python package for incomplete multi-view learn"
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = (
    "Alberto Lopez",
)
AUTHOR_EMAIL = "albertolz@proton.me"
URL = ""
DOC_URL = ""
MINIMUM_PYTHON_VERSION = "3.8"
with open("./requirements/base.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()
with open("./requirements/torch.txt", "r") as f:
    torch_extras = f.read()
with open("./requirements/r.txt", "r") as f:
    r_extras = f.read()
with open("./requirements/matlab.txt", "r") as f:
    matlab_extras = f.read()
EXTRA_PACKAGES = {
    'torch': torch_extras,
    'r': r_extras,
    'matlab': matlab_extras,
}

# Find version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, PACKAGE_NAME, "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < tuple([int(i) for i in MINIMUM_PYTHON_VERSION.split(".")]):
        sys.exit(f"Python {MINIMUM_PYTHON_VERSION}+ is required.")


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    url=URL,
    project_urls={
      'Documentation': DOC_URL,
      'Source': URL,
      'Tracker': URL + '/issues/',
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8"
        "Programming Language :: Python :: 3.9"
        "Programming Language :: Python :: 3.10"
        "Programming Language :: Python :: 3.11"
        "Programming Language :: Python :: 3.12"
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=f'>={MINIMUM_PYTHON_VERSION}',
)