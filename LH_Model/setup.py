from setuptools import setup

setup(
   name='lauritzen_hoffman',
   version='1.0',
   description='Package for computing secondary nucleation constants from DSC measurements Enthalpy vs time',
   author='Simona Buzzi',
   author_email='s.buzzi@tue.nl',
   packages=['lauritzen_hoffman'],  
   install_requires=['numpy', 'scipy', 'pandas', 'matplotlib'], #external packages as dependencies
    entry_points = {         
      'console_scripts': ['lauritzen_hoffman_fit=lauritzen_hoffman.linear_rate:lh_growth_model'],
  }
)