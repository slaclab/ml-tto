from importlib.metadata import PackageNotFoundError, version

try:
	from ._version import version as __version__
except ImportError:
	try:
		__version__ = version("ml_tto")
	except PackageNotFoundError:
		__version__ = "0.0.0"


__all__ = ["__version__"]
