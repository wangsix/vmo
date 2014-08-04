from distutils.core import setup

setup(name="vmo",
      version="0.1",
      description="vmo - Variable Markov Oracle in Python",
      author="Cheng-i Wang",
      author_email='chw160@ucsd.edu',
      url='',
      py_modules = [
                    'vmo',
                    'vmo.analysis',
                    'vmo.draw',
                    'vmo.generate',
                    'vmo.VMO',
                    'vmo.VMO.oracle',
                    ],
      long_description = 'vmo is a Python library for time series and symbolic sequence analysis/synthesis in the family of software built around the Factor Oracle and Variable Markov Oracle algorithms. One of the main innovations in vmo is using functions related to Information Dynamics to determine oracle structure and query-matching algorithms.',
      )
