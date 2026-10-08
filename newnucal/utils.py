import jax.numpy as jnp
import numpy as np

# Compatibility: jax-finufft <= 1.3.1 tests ``x is batching.not_mapped``.  JAX
# 0.10 removed that alias; it was always ``None``, which is what JAX 0.10 uses
# for an unmapped batch dimension.  Restore it if missing (no-op on older JAX).
from jax._src.interpreters import batching as _batching
if not hasattr(_batching, "not_mapped"):
    _batching.not_mapped = None

if False:
    DTYPE_R_JAX = jnp.float32
    DTYPE_R_NPY =  np.float32
    DTYPE_C_JAX = jnp.complex64
    DTYPE_C_NPY =  np.complex64
else:
    DTYPE_R_JAX = jnp.float64
    DTYPE_R_NPY =  np.float64
    DTYPE_C_JAX = jnp.complex128
    DTYPE_C_NPY =  np.complex128

try:
    from fftvis.utils import speed_of_light as _C_import
    C = DTYPE_R_NPY(_C_import)  # m / s
except ImportError:
    C = DTYPE_R_NPY(299_792_458.0)  # m / s
