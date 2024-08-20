from setuptools import setup

setup(
    name='imvc',
    version='0.1.0',
    packages=["imvc"],
    install_requires=[
        "scikit-learn==1.2.2",
        "pandas==2.0.1",
        "numpy==1.23.5",
        "scipy==1.10.1",
        "networkx==3.1",
        "gensim==4.3.1",
        "h5py==3.8.0",
        "snfpy==0.2.2",
        "oct2py==5.6.0",
        "rpy2==3.5.14",
    ],
    author='Alberto Lopez',
    author_email='albertolz@proton.me',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
)