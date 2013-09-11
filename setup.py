from distutils.core import setup


import aclust

# Run 2to3 builder if we're on Python 3.x, from
#   http://wiki.python.org/moin/PortingPythonToPy3k
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py
command_classes = {'build_py': build_py}

setup(name='aclust',
      version=aclust.__version__,
      description=\
        "streaming agglomerative clustering",
      url="https://github.com/brentp/aclust/",
      py_modules=['aclust'],
      long_description=open('README.md').read(),
      platforms='any',
      classifiers=[
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
          ],
      keywords='bioinformatics cluster',
      author='brentp',
      author_email='bpederse@gmail.com',
      license='MIT',
      include_package_data=True,
      tests_require=['nose'],
      test_suite='nose.collector',
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      scripts=[],
      entry_points={
       #'console_scripts': ['pyfasta = pyfasta:main']
      },
      cmdclass=command_classes,
  )
