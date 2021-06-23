from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
      name='asdfghjkl',
      version='0.0.1',
      description='ASDL: Automatic Second-order Differentiation (for Fisher, Gradient covariance, Hessian, Jacobian, and Kernel) Library',
      install_requires=requirements,
      packages=find_packages(),
      python_requires='>=3.6'
      )
