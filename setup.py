from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()



setup(
    name='acpreprocessing',


    description='Axonal connectomics processing',
    long_description='Tools for processing axonal connectomics datasets',

    # Author details
    author='Sharmishtaa Seshamani',
    author_email='',

    #requirements
    install_requires=required,

    # Choose your license
    license='Allen Institute Sofware License',
    packages=find_packages()
)
