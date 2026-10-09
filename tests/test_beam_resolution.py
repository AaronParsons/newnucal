"""ForwardModel warns when the beam grid is coarser than 4x the sky grid.

Bilinear beam interpolation onto the rotating sky pixels kinks at beam-pixel
crossings and puts power at all fringe rates at in-horizon delays (specred
memo-002).  The check warns rather than raises, so coarse beams stay usable.
"""

import warnings

import numpy as np
import pytest

from newnucal import BeamModel, SkyModel, ForwardModel, Calibrator, BeamResolutionWarning
from newnucal.simulate import check_beam_resolution


@pytest.mark.parametrize("sky_nside, beam_nside, ok", [
    (8, 8, False), (8, 16, False), (8, 31, False), (8, 32, True), (64, 256, True),
    (64, 128, False),
])
def test_check_beam_resolution(sky_nside, beam_nside, ok):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert check_beam_resolution(sky_nside, beam_nside) is ok
    hits = [w for w in rec if issubclass(w.category, BeamResolutionWarning)]
    assert len(hits) == (0 if ok else 1)
    if not ok:
        assert "memo-002" in str(hits[0].message)


def test_forward_model_warns_for_coarse_beam(array, freqs, sky_basis, beam_basis):
    sky = SkyModel(nside=8, freqs=freqs, basis=sky_basis)
    beam = BeamModel(nside=8, freqs=freqs, basis=beam_basis)
    with pytest.warns(BeamResolutionWarning, match="beam nside 8 is below 4 x sky nside 8"):
        ForwardModel(array, sky, beam, freqs)


def test_forward_model_silent_for_fine_beam(array, freqs, sky_basis, beam_basis):
    sky = SkyModel(nside=8, freqs=freqs, basis=sky_basis)
    beam = BeamModel(nside=32, freqs=freqs, basis=beam_basis)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BeamResolutionWarning)
        ForwardModel(array, sky, beam, freqs)


def test_calibrator_warns_once_not_per_simulate(array, freqs, rot_matrices, sky_model,
                                                beam_model):
    """Fires at construction only; simulate() and loss evaluations stay quiet."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        nt, nb = rot_matrices.shape[0], array.bls.shape[0]
        cal = Calibrator(array=array, beam_model=beam_model, sky_model=sky_model,
                         freqs=freqs, rot_matrices=rot_matrices,
                         data=np.zeros((nt, freqs.size, nb), dtype=complex))
        n_init = sum(issubclass(w.category, BeamResolutionWarning) for w in rec)
        prm = cal.init_params()
        cal.simulate(prm)
        cal.simulate(prm)
        n_after = sum(issubclass(w.category, BeamResolutionWarning) for w in rec)
    assert n_init == 1
    assert n_after == 1
