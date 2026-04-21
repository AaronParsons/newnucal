"""
HERA array geometry.

HERAArray wraps fftvis antenna gridding to produce the three things needed
by the forward model:

  basis_matrix  [3, 3]      Converts topocentric unit vectors to grid
                            coordinates.  Has been divided by c so that
                            basis_matrix @ topo_uvec * freq gives
                            dimensionless NUFFT arguments (cycles, not
                            radians — the factor of 2π is applied later).

  bl_grid       [nbls, 2]   Integer (East, North) grid positions for each
                            unique baseline.  Positive and negative values
                            are both used; Python negative indexing into
                            the NUFFT output grid gives the correct mode.

  n_modes       int         NUFFT grid edge length, chosen to contain all
                            baselines.
"""

import numpy as np
import hera_sim.antpos
import fftvis.core.antenna_gridding as _fg
import fftvis.utils as _fu
from .utils import DTYPE_R_NPY as DTYPE_R
from .utils import C
from . import hexrect

class HERAArray:
    """
    Geometric description of a HERA-like hexagonal array.

    Parameters
    ----------
    ants : dict[int, array_like]
        Antenna positions in metres, keyed by antenna index.
        Each value is a 3-element (East, North, Up) position.
    """

    def __init__(self, ants: dict):
        self.ants = {k: np.asarray(v, dtype=DTYPE_R) for k, v in ants.items()}
        antpos = np.array(list(self.ants.values()), dtype=DTYPE_R)

        # Unique baselines — one representative per redundant group
        red_gps = _fu.get_pos_reds(self.ants)
        self.antpairs = np.array([gp[0] for gp in red_gps], dtype=int)  # (nbls, 2)
        self.bls = np.array(
            [self.ants[i] - self.ants[j] for i, j in self.antpairs],
            dtype=DTYPE_R,
        )  # (nbls, 3)  metres

        # Grid baseline positions and change-of-basis matrix
        _, gridded_antpos, basis_matrix = (
            _fg.check_antpos_griddability(self.ants)
        )
        # bl_grid[bl, 0/1]: integer (East, North) grid positions
        self.bl_grid = np.array(
            [
                np.round(gridded_antpos[j] - gridded_antpos[i]).astype(int)[:2]
                for i, j in self.antpairs
            ],
            dtype=int,
        )  # (nbls, 2)

        self.n_modes = int(2 * np.max(np.abs(self.bl_grid)) + 1)

        # Divide basis_matrix by c so the NUFFT argument is
        #   x[freq, px] = 2π * (basis_matrix @ topo)[0, px] * freq
        # where topo are dimensionless direction cosines.
        self.basis_matrix = (basis_matrix / C).astype(DTYPE_R)  # (3, 3)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_hex(cls, hexnum: int, sep: float = 14.6, outriggers: bool = False):
        """
        Build a HERAArray from a hex array specification.

        Parameters
        ----------
        hexnum : int
            Number of antennas along one side of the hex core.
        sep : float
            Antenna separation in metres.
        outriggers : bool
            Include outrigger antennas.
        """
        hx = hera_sim.antpos.HexArray(sep=sep, outriggers=outriggers)
        ants = hx(hexnum, outriggers=outriggers)
        return cls(ants)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nants(self) -> int:
        return len(self.ants)

    @property
    def nbls(self) -> int:
        return self.antpairs.shape[0]

    @property
    def bmax(self) -> float:
        """Maximum baseline length (m) in the East-North plane."""
        return float(np.max(np.linalg.norm(self.bls[:, :2], axis=1)))

    @property
    def critical_channel_spacing(self) -> float:
        """Minimum channel spacing (Hz) for alias-free 2D NUFFT path.

        Uses the longest baseline length and default parameters (field_radius=1.0, safety=1.0).
        """
        return hexrect.critical_channel_spacing(self.bmax)

    def freq_array(self, freq_min: float, freq_max: float, field_radius: float = 1.0, safety: float = 1.0) -> np.ndarray:
        """Generate frequency array with critical channel spacing.

        Parameters
        ----------
        freq_min : float
            Minimum frequency (Hz).
        freq_max : float
            Maximum frequency (Hz).
        field_radius : float, optional
            Maximum sky radius in direction cosines (1.0 = full horizon). Default 1.0.
        safety : float, optional
            Safety factor to increase channel density (> 1). Default 1.0.

        Returns
        -------
        np.ndarray
            Frequency array from freq_min to freq_max with critical spacing.
        """
        dnu = hexrect.critical_channel_spacing(self.bmax, field_radius=field_radius, safety=safety)
        nfreqs = int(np.ceil((freq_max - freq_min) / dnu)) + 1
        return np.linspace(freq_min, freq_max, nfreqs, dtype=DTYPE_R)
