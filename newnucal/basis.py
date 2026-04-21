"""
Spectral basis classes for beam and sky models.

:class:`BeamBasis` and :class:`SkyBasis` each store a pre-built orthonormal
spectral basis matrix ``A`` (nfreq, nmodes) along with optional SVD metadata.

Construction paths
------------------
1. DPSS          : ``BeamBasis.from_dpss(freqs, eta_max)`` — no SVD metadata
2. File          : ``BeamBasis.from_file(path)`` — loads from ``.npz``
3. Ensemble      : ``BeamBasis.from_ensemble(freqs, spectra, n_modes)``
4. Beam diameters: ``BeamBasis.from_beam_diameters(freqs, diameters, ...)``

File format (``.npz``)
----------------------
``A``          (nfreq, nmodes)           — orthonormal basis matrix
``freqs_hz``   (nfreq,) float64          — frequency array (optional)
``svd_mean``   (nfreq,)                  — mean spectrum (optional)
``svd_modes``  (n_modes, nfreq)          — raw SVD row-vectors (optional)
``svd_svals``  (n_modes,)                — singular values (optional)
``n_samples``  scalar int64              — ensemble size (optional)
"""

import numpy as np
from .dpss import dpss_matrix
from .utils import DTYPE_R_NPY as DTYPE_R


class _SpectralBasis:
    """Base class for spectral basis objects.

    Parameters
    ----------
    A : array_like, (nfreq, nmodes)        
        Orthonormal basis matrix.
    freqs_hz : array_like, (nfreq,) float64, optional
    svd_mean : array_like, (nfreq,)        , optional
        Mean spectrum used to build ``A``.
    svd_modes : array_like, (n_modes, nfreq)        , optional
        Raw SVD row-vectors (before QR orthonormalisation).
    svd_svals : array_like, (n_modes,)        , optional
        Singular values corresponding to ``svd_modes``.
    n_samples : int, optional
        Number of ensemble samples from which SVD was derived.
    """

    def __init__(
        self,
        A,
        freqs_hz=None,
        svd_mean=None,
        svd_modes=None,
        svd_svals=None,
        n_samples=None,
    ):
        self.A        = np.asarray(A, dtype=DTYPE_R)
        self.freqs_hz = np.asarray(freqs_hz, dtype=DTYPE_R) if freqs_hz is not None else None
        self.svd_mean  = np.asarray(svd_mean,  dtype=DTYPE_R) if svd_mean  is not None else None
        self.svd_modes = np.asarray(svd_modes, dtype=DTYPE_R) if svd_modes is not None else None
        self.svd_svals = np.asarray(svd_svals, dtype=DTYPE_R) if svd_svals is not None else None
        self.n_samples = int(n_samples) if n_samples is not None else None

    def project(self, data):
        """Project data (npix, nfreq) onto internal basis. Return coefficients."""
        return data @ self.A

    def deproject(self, coeffs):
        """De-project coeffs (npix, coeffs) back to frequency basis."""
        return coeffs @ self.A.T

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dpss(cls, freqs, eta_max, eigenval_cutoff: float = 1e-9):
        """Build a DPSS basis (no SVD metadata).

        Parameters
        ----------
        freqs : array_like, (nfreq,) Hz
        eta_max : float
            DPSS delay half-width in seconds.
        eigenval_cutoff : float
            Eigenvalue cutoff for mode selection.
        """
        freqs = np.asarray(freqs, dtype=np.float64)
        A = dpss_matrix(freqs, eta_max, eigenval_cutoff=eigenval_cutoff)
        return cls(A=A, freqs_hz=freqs)

    @classmethod
    def from_ensemble(cls, freqs, spectra, n_modes: int):
        """Derive an eigenbasis from an ensemble of spectra.

        Computes the covariance SVD of *spectra* and QR-orthonormalises
        ``[mean; Vt[:n_modes]]`` to produce the basis matrix ``A``.

        Parameters
        ----------
        freqs : array_like, (nfreq,) Hz
        spectra : array_like, (n_samples, nfreq)
            Ensemble rows (e.g. beam power or sky flux spectra).
        n_modes : int
            Number of SVD modes to retain (``A`` will have ``n_modes+1``
            columns: mean + modes).
        """
        freqs = np.asarray(freqs, dtype=np.float64)
        pows  = np.asarray(spectra, dtype=np.float64)
        n_samples, nfreq = pows.shape
        mean   = pows.mean(axis=0)               # (nfreq,)
        pows_c = pows - mean[None, :]
        C      = pows_c.T @ pows_c               # (nfreq, nfreq)
        _, svals, Vt = np.linalg.svd(C)          # svals: (nfreq,), Vt: (nfreq, nfreq)
        n_modes = min(n_modes, nfreq - 1)
        # QR-orthonormalize [mean; SVD modes] → A (nfreq, 1+n_modes)
        raw = np.vstack([mean[None, :], Vt[:n_modes]])  # (1+n_modes, nfreq)
        Q, _ = np.linalg.qr(raw.T)                      # (nfreq, 1+n_modes)
        A = Q.astype(DTYPE_R)
        return cls(
            A=A,
            freqs_hz=freqs,
            svd_mean=mean.astype(DTYPE_R),
            svd_modes=Vt[:n_modes].astype(DTYPE_R),
            svd_svals=svals[:n_modes].astype(DTYPE_R),
            n_samples=n_samples,
        )

    def save(self, path):
        """Save to a ``.npz`` file."""
        arrays = {'A': self.A}
        if self.freqs_hz  is not None: arrays['freqs_hz']  = self.freqs_hz
        if self.svd_mean  is not None: arrays['svd_mean']  = self.svd_mean
        if self.svd_modes is not None: arrays['svd_modes'] = self.svd_modes
        if self.svd_svals is not None: arrays['svd_svals'] = self.svd_svals
        if self.n_samples is not None: arrays['n_samples'] = np.array(self.n_samples, dtype=np.int64)
        np.savez(path, **arrays)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_svd(self) -> bool:
        """True when all SVD metadata is present."""
        return (
            self.svd_mean  is not None
            and self.svd_modes is not None
            and self.svd_svals is not None
            and self.n_samples is not None
        )

    @property
    def nfreq(self) -> int:
        return int(self.A.shape[0])

    @property
    def nmodes(self) -> int:
        return int(self.A.shape[1])

    # ------------------------------------------------------------------
    # Internal helpers for subclass from_file
    # ------------------------------------------------------------------

    @staticmethod
    def _load_arrays(path):
        return dict(np.load(path, allow_pickle=False))

    @classmethod
    def _from_npz(cls, d):
        return cls(
            A=d['A'],
            freqs_hz=d.get('freqs_hz'),
            svd_mean=d.get('svd_mean'),
            svd_modes=d.get('svd_modes'),
            svd_svals=d.get('svd_svals'),
            n_samples=int(d['n_samples']) if 'n_samples' in d else None,
        )


class BeamBasis(_SpectralBasis):
    """Spectral basis for beam models."""

    @classmethod
    def from_file(cls, path):
        """Load a :class:`BeamBasis` from a ``.npz`` file."""
        return cls._from_npz(cls._load_arrays(path))

    @classmethod
    def from_beam_diameters(
        cls,
        freqs,
        diameters,
        nside: int = 32,
        ch_chunk: int = 32,
        n_modes: int = 17,
    ):
        """Derive a beam eigenbasis from AiryBeam models at given dish diameters.

        Evaluates one AiryBeam per diameter on a HEALPix grid (above-horizon
        pixels only) and calls :meth:`from_ensemble` on the resulting power
        spectra.

        Parameters
        ----------
        freqs : array_like, (nfreq,) Hz
        diameters : sequence of float
            Dish diameters in metres.
        nside : int
            HEALPix resolution for beam evaluation.
        ch_chunk : int
            Frequency channels evaluated per call (memory control).
        n_modes : int
            SVD modes to retain; basis has ``n_modes+1`` columns.
        """
        import healpy as hp
        from pyuvdata.analytic_beam import AiryBeam

        freqs = np.asarray(freqs, dtype=np.float64)
        nfreq = len(freqs)
        npix  = hp.nside2npix(nside)
        th, ph = hp.pix2ang(nside, np.arange(npix))
        above  = th < np.pi / 2
        az = ph[above].astype(np.float64)
        za = th[above].astype(np.float64)

        def _eval(beam, az, za):
            n = len(az)
            pwr = np.empty((n, nfreq), dtype=DTYPE_R)
            for ch in range(0, nfreq, ch_chunk):
                sl = slice(ch, ch + ch_chunk)
                resp = beam.power_eval(az_array=az, za_array=za,
                                       freq_array=freqs[sl])[0, 0]
                pwr[:, sl] = resp.T.astype(DTYPE_R)
            return pwr  # (n_above, nfreq)

        rows = []
        for diam in diameters:
            rows.append(_eval(AiryBeam(diameter=float(diam), include_cross_pols=False), az, za))
        spectra = np.concatenate(rows, axis=0).astype(np.float64)
        return cls.from_ensemble(freqs, spectra, n_modes)


class SkyBasis(_SpectralBasis):
    """Spectral basis for sky models."""

    @classmethod
    def from_file(cls, path):
        """Load a :class:`SkyBasis` from a ``.npz`` file."""
        return cls._from_npz(cls._load_arrays(path))
