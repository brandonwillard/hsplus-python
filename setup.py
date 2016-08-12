from setuptools import setup, find_packages
#from Cython.Build import cythonize

version = __import__('hsplus').get_version()

#ext_modules = cythonize("hsplus/*.pyx")

setup(name='hsplus',
      version=version,
      description='HS+ Prior Code',
      long_description="Code for modeling with the HS+ prior.",
      url='https://bitbucket.com/bayes-horseshoe-plus/hsplus-python-pkg/',
      author='Brandon T. Willard',
      author_email='brandonwillard@gmail.com',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      keywords='mpmath, sympy, bayes, horseshoe prior',
      packages=find_packages(),
      install_requires=['scipy>=0.17.0',
                        'numpy>=1.10.4',
                        'mpmath>=0.19',
                        #'Cython>=0.23.4',
                        'sympy>=0.7.6',
                        'matplotlib',
                        ],
      #ext_modules=ext_modules,
      )
