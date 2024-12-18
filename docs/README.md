# Building Docs

We currently use Sphinx

## Instructions

### Python Dependencies

You will need to install all the dependencies as defined in `requirements.txt` file. The above can be installed
by entering:

    pip install -r requirements.txt

in the `docs/` directory.

## Generating the documentation

To build the HTML documentation, enter:

    make html

in the `docs/` directory. If all goes well, this will generate a `_build/html/` subdirectory containing the 
built documentation.