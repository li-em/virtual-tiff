"""OME-TIFFs as microscopy platforms write them, which this parser does not read yet.

Three published files, three distinct blockers. Each is a real object on a public host — the same
files a spatial-omics reader is handed — and each test asserts the *current* failure so that fixing
one turns its test red rather than leaving it silently passing.

* `IfdBig` — 10x's Xenium morphology images are BigTIFFs whose tags use the 64-bit IFD8 value type.
  `async_tiff` raises `RuntimeError: Unsupported value type 'IfdBig'` before any codec is consulted,
  so the JPEG 2000 support this parser already has is unreachable for them.
* Sub-IFDs — their H&E images carry the pyramid in SubIFDs rather than as top-level IFDs, and the
  parser declines with `NotImplementedError`. `ifd_layout="nested"` does not help: the levels are not
  IFDs of the file.
* Large objects over HTTP — a 17.7 GB uncompressed OME-TIFF fails while `async_tiff` reads the
  directory, with `Generic HTTP error: request or response body error`, reproducibly. Ranged GETs
  against the same object with a plain HTTP client succeed, so this is not the server refusing.

The files are 1.2 GB, 3.7 GB and 17.7 GB, and nothing here downloads them: a parser reads only the
directory.
"""

from __future__ import annotations

import pytest
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore

from virtual_tiff import VirtualTIFF

from .conftest import requires_network

TENX = "https://cf.10xgenomics.com/"
XENIUM_LUNG = TENX + "samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/"
AWS = "https://s3-us-west-2.amazonaws.com/"
ATERA = (AWS + "10x.files/samples/atera/dev/WTA_Preview_FFPE_Cervical_Cancer/"
         "WTA_Preview_FFPE_Cervical_Cancer_")


def parse(host: str, url: str, **kwargs):
    registry = ObjectStoreRegistry({host: HTTPStore(host)})
    return VirtualTIFF(**kwargs)(url, registry)


@requires_network
def test_bigtiff_with_ifd8_tag_values_is_refused():
    """Xenium morphology: a BigTIFF whose tag values use the IFD8 type.

    JPEG 2000 tiles (compression 33003/33005) are already in `COMPRESSORS`, so this file would be
    readable if the directory could be parsed at all.
    """
    with pytest.raises(RuntimeError, match="Unsupported value type 'IfdBig'"):
        parse(TENX, XENIUM_LUNG + "Xenium_V1_humanLung_Cancer_FFPE_morphology.ome.tif", ifd=0)


@requires_network
def test_pyramid_in_subifds_is_refused():
    """Xenium H&E: interleaved RGB whose resolution levels are SubIFDs.

    The parser handles `planar_configuration` 1 through `ChunkyCodec`, so only the pyramid's
    placement is in the way.
    """
    with pytest.raises(NotImplementedError, match="Sub-IFDs"):
        parse(TENX, XENIUM_LUNG + "Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif", ifd=0)


@requires_network
def test_reading_the_directory_of_a_17gb_object_over_http_fails():
    """Atera H&E: uncompressed, tiled, planar, ten levels, 17.7 GB.

    Every level is `planar_configuration` 2 with 1 MiB tiles, which this parser supports, and a
    plain HTTP client fetches any of those tiles by range. It is reading the *directory* that fails.
    """
    with pytest.raises(Exception, match="body error"):
        parse(AWS, ATERA + "he_image.ome.tif", ifd=0)
