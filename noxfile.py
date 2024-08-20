import nox

@nox.session(python=[
    "3.8",
    "3.9",
    "3.10",
    "3.11",
    "3.12",
])
def tests(session):
    session.install(
        "scikit-learn>=1.2.0",
        "pandas>=1.3.3",
        "numpy>=1.21.3,<2",
        "scipy>=1,<1.13",
        "networkx>=2.5",
        "gensim>=4.2",
        "h5py>=3.6",
        "snfpy>=0",
        # "oct2py>=5.5",
        # "rpy2==3.5.1",
        "pytest==8.3.2",
        # "coverage==7.6.0",
        # "pytest-cov==5.0.0",
    )
    session.run("pytest", "-v")
