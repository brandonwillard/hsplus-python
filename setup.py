from setuptools import setup, find_packages

version = __import__('hsplus').get_version()

setup(name='hsplus',
      version=version,
      description='Code for modeling with the HIB, HS and HS+ shrinkage priors.',
      long_description="Code for modeling with the HIB, HS and HS+ shrinkage priors.",
      url='https://bitbucket.com/bayes-horseshoe-plus/hsplus-python-pkg/',
      author='Brandon T. Willard',
      author_email='brandonwillard@gmail.com',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      keywords='mpmath, sympy, bayesian, shrinkage prior, regression',
      packages=find_packages(),
      setup_requires=['pytest-runner', ],
      tests_requires=['pytest', ],
      install_requires=[
                        'numpy>=1.10.4',
                        'mpmath>=0.19',
                        ],
      extra_require={
          'symbolic': ['sympy>=1.0',
                       ],
          'regression': ['scipy>=0.18.0',
                         'patsy>=0.4.1',
                         ],
      }
      )
