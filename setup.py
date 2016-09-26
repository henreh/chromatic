# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

DESCRIPTION = 'A library for algorithmic composition.'
DESCRIPTION_LONG = """A library for algorithmic composition. Chromatic provides
			a framework for combining the results of algorithmic generation and
			modification of music in a similar way to synths & effects in the 
			production process."""

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Environment :: Web Environment',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Multimedia :: Sound/Audio',
    'Topic :: Multimedia :: Sound/Audio :: MIDI',
    'Topic :: Artistic Software',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

if __name__ == '__main__':
    setuptools.setup(
        name='chromatic',
        version='0.0.1',
        description=DESCRIPTION,
        long_description=DESCRIPTION_LONG,
        author='Henry Franks',
        author_email='hpwfranks@googlemail.com',
        license='MIT',
        url='https://github.com/henreh/chromatic',
        classifiers=classifiers,
        packages=setuptools.find_packages(exclude=[]),
    )
