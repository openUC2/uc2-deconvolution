"""3D deconvolution for microscopy: PSF models, PSFs from beads, Richardson-Lucy and the solvers
of DeconvOptim.jl in numpy (``recipe``), and an HTTP worker that runs them for another machine
(``worker``)."""
from importlib.metadata import PackageNotFoundError, version

from .recipe import (Cancelled, Optics, Parameters, deconvolve, distill_psf, memory_bytes, psf,
                     run, synthetic)

try:
    __version__ = version("uc2-deconvolution")
except PackageNotFoundError:  # run from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = ["Cancelled", "Optics", "Parameters", "deconvolve", "distill_psf", "memory_bytes", "psf",
           "run", "synthetic", "__version__"]
