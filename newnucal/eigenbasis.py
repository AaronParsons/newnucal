"""
SkyBeamEigenbasis — spectral eigenbasis for the sky × beam product.

Derives, stores, and re-loads the compact spectral basis used by
:class:`~newnucal.grid_fitter.GridFitter`.  The basis is built from an
ensemble of sky flux spectra and beam power spectra; its product-product SVD
identifies the spectral modes that best represent the observable sky × beam
product space.

File format (``.npz``)
----------------------
``freqs_hz``       (nfreq,) float64
``beam_mean``      (nfreq,) float32   — mean beam power spectrum
``beam_modes``     (n_bm, nfreq) float32 — beam SVD modes (without mean)
``sky_mean``       (nfreq,) float32   — mean sky flux spectrum
``sky_modes``      (n_sk, nfreq) float32 — sky SVD modes (without mean)
``product_mean``   (nfreq,) float64   — mean row of the product-product matrix
``product_modes``  (n_pr, nfreq) float64 — SVD modes of centred product product
"""

import numpy as np
from .beam import BeamModel
from pyuvdata.analytic_beam import AiryBeam

DTYPE_R = np.float32


class SkyBeamEigenbasis:
    """Spectral eigenbasis for the sky × beam product.

    Parameters
    ----------
    freqs_hz : array_like, (nfreq,)
    beam_mean : array_like, (nfreq,)
        Mean beam power spectrum across the ensemble.
    beam_modes : array_like, (n_bm, nfreq)
        Beam SVD modes (not including the mean).
    sky_mean : array_like, (nfreq,)
        Mean sky flux spectrum across the ensemble.
    sky_modes : array_like, (n_sk, nfreq)
        Sky SVD modes (not including the mean).
    product_mean : array_like, (nfreq,)
        Mean row of the product-product matrix (used for centring).
    product_modes : array_like, (n_pr, nfreq)
        SVD modes of the centred product-product matrix.
    """

    def __init__(
        self,
        freqs_hz,
        beam_mean,
        beam_modes,
        sky_mean,
        sky_modes,
        product_mean,
        product_modes,
    ):
        self.freqs_hz    = np.asarray(freqs_hz,     dtype=np.float64)
        self.beam_mean   = np.asarray(beam_mean,    dtype=DTYPE_R)
        self.beam_modes  = np.asarray(beam_modes,   dtype=DTYPE_R)
        self.sky_mean    = np.asarray(sky_mean,     dtype=DTYPE_R)
        self.sky_modes   = np.asarray(sky_modes,    dtype=DTYPE_R)
        self.product_mean  = np.asarray(product_mean,  dtype=np.float64)
        self.product_modes = np.asarray(product_modes, dtype=np.float64)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_beam_sky_ensemble(
        cls,
        freqs,
        sky_fluxes,
        beam_powers=None,
        beam_diameters=None,
        nside: int = 32,
        n_beam_modes: int = 17,
        n_sky_modes: int = 5,
        n_product_modes: int = 20,
    ):
        """Derive eigenbases from an ensemble of sky fluxes and beam models.

        The product-product SVD follows the approach from the user's notebook:

        1. Build a beam power ensemble across HEALPix pixels and beam variants.
        2. Compute the mean and SVD of the covariance matrix.
        3. Repeat for the sky flux ensemble.
        4. Form all pairwise products of (mean + modes) weighted by their
           singular values; compute the SVD of the centred product-product
           matrix to get the product spectral modes.

        Parameters
        ----------
        freqs : array_like, (nfreq,)
            Frequency array in Hz.
        sky_fluxes : array_like, (n_sky_samples, nfreq)
            Ensemble of sky flux spectra.
        beam_models : array_like, (n_points, nfreq), optional
            Ensemble of beam powers.
        beam_diameters : array_like, optional
            Dish diameters in metres.  One beam per diameter is built.
            Provide this **or** ``beam_models``, not both.
        nside : int
            HEALPix nside for beam evaluation (used with ``beam_diameters``).
        n_beam_modes : int
            Number of SVD beam modes to retain (not counting the mean).
        n_sky_modes : int
            Number of SVD sky modes to retain (not counting the mean).
        n_product_modes : int
            Number of SVD product modes to retain.
        """
        import healpy as hp

        freqs = np.asarray(freqs, dtype=np.float64)
        nfreq = len(freqs)

        # ---- Beam ensemble ----------------------------------------
        if beam_powers is not None and beam_diameters is not None:
            raise ValueError("Provide beam_powers or beam_diameters, not both.")

        if beam_powers is not None:
            # Stack reconstructed beam spectra from pre-built array
            bmpwrs = np.asarray(beam_powers, dtype=np.float64) # (npix, nfreq)

        else:
            if beam_diameters is None:
                raise ValueError("Provide beam_models or beam_diameters.")

            # Build template BeamModel for the pixel grid
            tmpl = BeamModel(nside=nside, freqs=freqs, eta_max=beam_eta_max)
            npix = hp.nside2npix(tmpl.nside)
            th, ph = hp.pix2ang(tmpl.nside, np.arange(npix))
            # Evaluate only above-horizon pixels (za < π/2)
            above = th < np.pi / 2
            az = ph[above].astype(np.float64)
            za = th[above].astype(np.float64)

            def eval_beam(beam, az, za):
                npix = len(az)
                # power_eval returns an array of shape (1, Naxes_vec, nfreq_chunk, npix)
                resp = beam.power_eval(
                    az_array=az,
                    za_array=za,
                    freq_array=freqs,
                )[0, 0]  # → (nfreq_chunk, npix)
                beam_power = resp.T.astype(DTYPE_R)
                return beam_power

            rows = []
            for diam in beam_diameters:
                beam = AiryBeam(diameter=float(diam), include_cross_pols=False)
                rows.append(eval_beam(beam, az, za))
            bmpwrs = np.concatenate(rows, axis=0, dtype=DTYPE_R)  # (n_pix_total, nfreq)

        # Beam SVD
        beam_mean = bmpwrs.mean(axis=0, keepdims=True)  # (1, nfreq)
        beam_c    = bmpwrs - beam_mean
        C_bm      = np.dot(beam_c.T, beam_c)            # (nfreq, nfreq)
        _, S_bm, Vt_bm = np.linalg.svd(C_bm)
        # Extended: prepend mean (weight = n_samples)
        Vt_bm_ext = np.vstack([beam_mean, Vt_bm[:n_beam_modes]])  # (1+n_bm, nfreq)
        S_bm_ext  = np.concatenate([[float(bmpwrs.shape[0])], S_bm[:n_beam_modes]])

        # ---- Sky ensemble ----------------------------------------
        fluxes    = np.asarray(sky_fluxes, dtype=np.float64)
        flux_mean = fluxes.mean(axis=0, keepdims=True)   # (1, nfreq)
        flux_c    = fluxes - flux_mean
        C_fl      = np.dot(flux_c.T, flux_c)              # (nfreq, nfreq)
        _, S_fl, Vt_fl = np.linalg.svd(C_fl)
        Vt_fl_ext = np.vstack([flux_mean, Vt_fl[:n_sky_modes]])  # (1+n_sk, nfreq)
        S_fl_ext  = np.concatenate([[float(fluxes.shape[0])], S_fl[:n_sky_modes]])

        # ---- Outer product SVD ------------------------------------
        # product[k_bm, k_sk, f] = S_bm[k_bm] * Vt_bm[k_bm, f]
        #                       * S_fl[k_sk] * Vt_fl[k_sk, f]
        product = (
            S_bm_ext[:, None, None] * Vt_bm_ext[:, None, :]
            * S_fl_ext[None, :, None] * Vt_fl_ext[None, :, :]
        )  # (1+n_bm, 1+n_sk, nfreq)
        product = product.reshape(-1, nfreq)           # (prod_rows, nfreq)
        product_mean = product.mean(axis=0)             # (nfreq,)
        _, _, Vt_pr = np.linalg.svd(product - product_mean[None, :])

        return cls(
            freqs_hz      = freqs,
            beam_mean     = beam_mean[0].astype(DTYPE_R),
            beam_modes    = Vt_bm[:n_beam_modes].astype(DTYPE_R),
            sky_mean      = flux_mean[0].astype(DTYPE_R),
            sky_modes     = Vt_fl[:n_sky_modes].astype(DTYPE_R),
            product_mean  = product_mean,
            product_modes = Vt_pr[:n_product_modes],
        )

    @classmethod
    def from_file(cls, path):
        """Load a basis from a ``.npz`` file saved with :meth:`save`."""
        d = np.load(path)
        return cls(
            freqs_hz      = d['freqs_hz'],
            beam_mean     = d['beam_mean'],
            beam_modes    = d['beam_modes'],
            sky_mean      = d['sky_mean'],
            sky_modes     = d['sky_modes'],
            product_mean  = d['product_mean'],
            product_modes = d['product_modes'],
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path):
        """Save to a ``.npz`` file."""
        np.savez(
            path,
            freqs_hz      = self.freqs_hz,
            beam_mean     = self.beam_mean,
            beam_modes    = self.beam_modes,
            sky_mean      = self.sky_mean,
            sky_modes     = self.sky_modes,
            product_mean  = self.product_mean,
            product_modes = self.product_modes,
        )

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------

    def orthonormal_sky_basis(self) -> np.ndarray:
        """Return QR-orthonormalized sky basis as ``(nfreq, K)`` float32.

        Prepends ``sky_mean`` to ``sky_modes`` and QR-orthonormalizes.
        The returned matrix has shape ``(nfreq, K)`` matching the ``A_sky``
        convention used by :class:`~newnucal.simulate.ForwardModel` and
        :class:`~newnucal.calibrator.Calibrator`:
        ``sky_spec = sky_coeffs @ A_sky.T``.
        """
        raw = np.vstack([self.sky_mean[None, :], self.sky_modes])  # (1+n_sk, nfreq)
        Q, _ = np.linalg.qr(raw.T)    # (nfreq, K)
        return Q.astype(DTYPE_R)       # (nfreq, K)

    def orthonormal_beam_basis(self) -> np.ndarray:
        """Return QR-orthonormalized beam basis as ``(nfreq, K)`` float32.

        Prepends ``beam_mean`` to ``beam_modes`` and QR-orthonormalizes.
        The returned matrix has shape ``(nfreq, K)`` matching the ``A_beam``
        convention used by :class:`~newnucal.beam.BeamModel` and
        :class:`~newnucal.simulate.ForwardModel`:
        ``beam_spec = beam_coeffs @ A_beam.T``.
        """
        raw = np.vstack([self.beam_mean[None, :], self.beam_modes])  # (1+n_bm, nfreq)
        Q, _ = np.linalg.qr(raw.T)    # (nfreq, K)
        return Q.astype(DTYPE_R)       # (nfreq, K)

    def orthonormal_product_basis(self) -> np.ndarray:
        """Return QR-orthonormalized product basis as ``(K, nfreq)`` float32.

        Prepends the mean spectrum to the SVD modes, then orthonormalizes
        via QR decomposition.  The resulting rows form an orthonormal basis
        that spans the same space as ``[product_mean, product_modes]``.

        .. note::
            This returns ``(K, nfreq)`` — *transposed* relative to
            :meth:`orthonormal_sky_basis` and :meth:`orthonormal_beam_basis`
            — to match the row-major convention used by
            :class:`~newnucal.grid_fitter.GridFitter`.
        """
        raw = np.vstack([self.product_mean[None, :], self.product_modes])
        Q, _ = np.linalg.qr(raw.T)   # (nfreq, K)
        return Q.T.astype(DTYPE_R)    # (K, nfreq)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nfreq(self) -> int:
        return int(self.freqs_hz.shape[0])

    @property
    def n_beam_modes(self) -> int:
        return int(self.beam_modes.shape[0])

    @property
    def n_sky_modes(self) -> int:
        return int(self.sky_modes.shape[0])

    @property
    def n_product_modes(self) -> int:
        return int(self.product_modes.shape[0])

    @property
    def n_sky_basis(self) -> int:
        """Number of orthonormal sky basis vectors (including mean)."""
        return self.n_sky_modes + 1

    @property
    def n_beam_basis(self) -> int:
        """Number of orthonormal beam basis vectors (including mean)."""
        return self.n_beam_modes + 1

    @property
    def nbasis(self) -> int:
        """Number of orthonormal product basis vectors (including mean)."""
        return self.n_product_modes + 1
