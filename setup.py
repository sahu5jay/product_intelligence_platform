from setuptools import setup, find_packages
from typing import List

def get_requirements(filepath : str)-> List[str]:

    requirements = []
    with open(filepath) as file:
        requirements = file.readLines()
        requirements = [req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name="product_intelligence_platform",
    version="0.0.1",
    author="Jay Sahu",
    author_email="jsahu0523xample.com",
    description="Multi-Modal AI Product Intelligence Platform",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)