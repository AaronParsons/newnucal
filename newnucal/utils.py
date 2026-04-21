import jax.numpy as jnp
import numpy as np

DTYPE_R_JAX = jnp.float32
DTYPE_R_NPY =  np.float32
DTYPE_C_JAX = jnp.complex64
DTYPE_C_NPY =  np.complex64

try:
    from fftvis.utils import speed_of_light as _C_import
    C = DTYPE_R_NPY(_C_import)  # m / s
except ImportError:
    C = DTYPE_R_NPY(299_792_458.0)  # m / s
