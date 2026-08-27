# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Consolidated metadata is optional and store-aware (gh-issue-698).

Icechunk rejects consolidated metadata because it interferes with its own
consolidation and consistency mechanisms, so ``to_ngff_zarr`` must be able to
skip it -- either because the store advertises no support or because the caller
asks for it explicitly.
"""

import numpy as np
import packaging.version
import pytest
import zarr
from ngff_zarr import from_ngff_zarr, to_multiscales, to_ngff_image, to_ngff_zarr
from zarr.storage import MemoryStore

zarr_version_major = packaging.version.parse(zarr.__version__).major

# supports_consolidated_metadata is a Zarr v3 store concept.
pytestmark = pytest.mark.skipif(
    zarr_version_major < 3,
    reason="supports_consolidated_metadata is a Zarr v3 store concept",
)


class NoConsolidateStore(MemoryStore):
    """A store that reports no consolidated-metadata support, like Icechunk."""

    @property
    def supports_consolidated_metadata(self):
        return False


def _make_multiscales():
    data = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    image = to_ngff_image(data, dims=["y", "x"])
    return to_multiscales(image, scale_factors=[2])


def _is_consolidated(store):
    group = zarr.open_group(store, mode="r")
    return group.metadata.consolidated_metadata is not None


def test_consolidates_by_default():
    store = MemoryStore()
    to_ngff_zarr(store, _make_multiscales(), version="0.5")
    assert _is_consolidated(store)


def test_skips_when_store_reports_no_support():
    store = NoConsolidateStore()
    to_ngff_zarr(store, _make_multiscales(), version="0.5")
    assert not _is_consolidated(store)


def test_explicit_false_skips_on_supporting_store():
    store = MemoryStore()
    to_ngff_zarr(store, _make_multiscales(), version="0.5", consolidate_metadata=False)
    assert not _is_consolidated(store)


def test_unconsolidated_store_roundtrips():
    store = NoConsolidateStore()
    multiscales = _make_multiscales()
    to_ngff_zarr(store, multiscales, version="0.5")

    roundtrip = from_ngff_zarr(store)
    np.testing.assert_array_equal(
        np.asarray(roundtrip.images[0].data),
        np.asarray(multiscales.images[0].data),
    )


def test_ozx_honors_explicit_false(tmp_path):
    """The .ozx zipped path forwards the override rather than always consolidating."""
    path = tmp_path / "image.ozx"
    to_ngff_zarr(
        str(path), _make_multiscales(), version="0.5", consolidate_metadata=False
    )

    store = zarr.storage.ZipStore(str(path), mode="r")
    assert not _is_consolidated(store)
