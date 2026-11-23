"""The HTTP worker: a job gives the local result, failures are reported, cancel stops a job."""
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from uc2deconvolution import recipe, worker

SPEC = {"optics": {"pixel_um": 0.12, "z_step_um": 0.5}, "parameters": {"iterations": 4}}


@pytest.fixture
def worker_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), worker.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_job_gives_the_local_result(worker_url):
    _, stack = recipe.synthetic((8, 96, 96), recipe.Optics(pixel_um=0.12, z_step_um=0.5))
    stack = stack.astype(np.uint16)
    planes = list(worker.run_remote(worker_url, iter(stack), stack.shape, stack.dtype, SPEC,
                                    poll_s=0.01))
    np.testing.assert_allclose(np.stack(planes), recipe.run(stack, SPEC), rtol=1e-5)
    assert worker.info(worker_url)["recipes"] == {recipe.NAME: recipe.VERSION}


def test_worker_reports_failures_and_cancels(worker_url):
    volume = np.ones((4, 32, 32), np.uint16)
    bad = {"optics": {"pixel_um": 0.12, "na": 1.1}}
    with pytest.raises(RuntimeError, match="NA 1.1"):
        list(worker.run_remote(worker_url, volume, volume.shape, volume.dtype, bad, poll_s=0.01))
    slow = {"optics": {"pixel_um": 0.12}, "parameters": {"iterations": 500}}
    big = np.ones((8, 128, 128), np.uint16)  # still running at the first poll, however busy
    with pytest.raises(recipe.Cancelled):
        list(
            worker.run_remote(
                worker_url, big, big.shape, big.dtype, slow, progress=lambda f: True, poll_s=0.01
            )
        )
    with pytest.raises(RuntimeError, match="unknown recipe"):
        list(worker.run_remote(worker_url, volume, volume.shape, volume.dtype, slow, recipe="nope"))
    assert worker.info(worker_url)["jobs"] == 0  # every job above was deleted
