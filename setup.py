from distutils.core      import setup
from distutils.extension import Extension
from Cython.Build        import cythonize


setup(name="divreg",
	version='0.1.0',
	packages=["divreg"],
	package_dir={'divreg': 'src'},
	author='Viktor Reshniak',
	author_email='reshniakv@ornl.gov'
	)