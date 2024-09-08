from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'Package/Library to do Exploratory Data Analysis'
LONG_DESCRIPTION = 'A package that allows to build Exploratory Data Analysis just by passing Dataset and column names.'

def get_requirements(file_path: str):
    """
    this function will return the list of libraries in the requirements.txt file

    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    return requirements

# Setting up
setup(
    name="Exploratory-Data-Analysis",
    version=VERSION,
    author="Karthik",
    author_email="<karthiksurya611@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)