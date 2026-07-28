import copy
import json
import packaging.version

import imageio
import imageio.plugins.tifffile
import numpy
import tifffile

from imageio.config import (
    FileExtension,
    PluginConfig,
    known_extensions,
    known_plugins,
)


class ACTiffFormat(imageio.plugins.tifffile.TiffFormat):
    """custom TIFF format through imageio.v2 compatibility API"""


    MICRO_MANAGER_TAG = 51123
    MICRO_MANAGER_KEY = "MicroManagerMetadata"

    class Reader(imageio.plugins.tifffile.TiffFormat.Reader):
        _acquisition_md_tags = [
            51123,
            "MicroManagerMetadata"
            ]

        def _open(self, multifile=False, **kwargs):
            self._tf_multifile = multifile
            if (packaging.version.parse(
                    tifffile.__version__) >= packaging.version.parse(
                        "2020.9.30")):
                kwargs["_multifile"] = multifile
            else:
                kwargs["multifile"] = multifile
            return super()._open(**kwargs)

        def _get_meta_data(self, index):
            meta = super()._get_meta_data(index)

            series_index = 0 if index is None else index
            page = self._tf.series[series_index].pages[0]

            for tag_key in (
                ACTiffFormat.MICRO_MANAGER_TAG,
                ACTiffFormat.MICRO_MANAGER_KEY,
            ):
                try:
                    value = page.tags[tag_key].value
                except KeyError:
                    continue

                if isinstance(value, bytes):
                    value = value.rstrip(b"\x00").decode("utf-8")

                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass

                meta[ACTiffFormat.MICRO_MANAGER_KEY] = value
                break

            return meta

        def _get_data(self, index):
            if (
                self.request.mode[1] in "vV"
                and not self._tf_multifile
            ):
                pages = self._tf.series[index].pages

                first = pages[0].asarray()
                image = numpy.empty(
                    (len(pages), *first.shape),
                    dtype=first.dtype,
                )
                image[0] = first

                for page_index, page in enumerate(
                    pages[1:],
                    start=1,
                ):
                    image[page_index] = page.asarray()
            else:
                image, _ = super()._get_data(index)

            return image, self._get_meta_data(index)

        def _get_data_legacy(self, index):
            # reading as a volume and ignoring multifile is faster this way
            #   and avoids tifffile log messages
            if self.request.mode[1] in "vV" and not self._tf_multifile:
                for i, p in enumerate(self._tf.series[index].pages):
                    img = p.asarray()
                    if i == 0:
                        im = numpy.empty(
                            (len(self._tf.series[index].pages),
                             *img.shape),
                            dtype=img.dtype)
                    im[i, ...] = img
            else:
                im, meta = super()._get_data(index)
            return im, self._get_meta_data(index or 0)

    class Writer(imageio.plugins.tifffile.TiffFormat.Writer):
        _description_encoding = "latin-1"

        def _append_data(self, image, meta):
            # Avoid modifying the caller's dictionary.
            meta = dict(meta or {})

            mm_metadata = None
            for key in (
                ACTiffFormat.MICRO_MANAGER_KEY,
                ACTiffFormat.MICRO_MANAGER_TAG,
            ):
                if key in meta:
                    mm_metadata = meta[key]
                    break

            if mm_metadata is not None:
                extratags = list(meta.get("extratags", ()))
                extratags.append(
                    (
                        ACTiffFormat.MICRO_MANAGER_TAG,
                        "s",
                        0,
                        json.dumps(mm_metadata),
                        True,
                    )
                )
                meta["extratags"] = extratags

            # Prevent ZYX data with Z=3 or Z=4 from being inferred as RGB.
            if image.ndim == 3:
                meta.setdefault("photometric", "minisblack")
            elif image.ndim == 4 and image.shape[-1] in (3, 4):
                meta.setdefault("photometric", "rgb")

            description = meta.get("description")
            if isinstance(description, str):
                meta["description"] = description.encode(
                    self._description_encoding
                )

            return super()._append_data(image, meta)



def filter_metadata(meta, fields_to_keep):
    return {
        key: copy.deepcopy(meta[key])
        for key in fields_to_keep
        if key in meta
    }


def volwrite_preserving_md_fields(uri, volume, *, metadata_fields=None, **kwargs):
    metadata_fields = metadata_fields or tuple()

    source_metadata = getattr(volume, "meta", None)
    write_metadata = filter_metadata(source_metadata, metadata_fields)

    return imageio.v2.volwrite(
        uri,
        numpy.asarray(volume),
        metadata=write_metadata,
        **kwargs
    )

# fmt = ACTiffFormat(
#     "actiff",
#     "tiff format for AIBS axonal connectomics",
#     ".actiff .tiff .tif",
#     "iIvV"
# )

# imageio.formats.add_format(fmt, overwrite=True)


def register_actiff() -> None:
    """Register ACTIFF in ImageIO's legacy-plugin compatibility registry."""

    plugin_name = "ACTIFF"

    known_plugins[plugin_name] = PluginConfig(
        name=plugin_name,
        class_name=ACTiffFormat.__name__,
        module_name=__name__,
        is_legacy=True,
        install_name="tifffile",
        legacy_args={
            "description": (
                "TIFF format for AIBS axonal connectomics"
            ),
            "extensions": ".actiff .tiff .tif",
            "modes": "iIvV",
        },
    )

    # Claim only the custom extension automatically. For .tif/.tiff,
    # explicitly specify format="ACTIFF".
    extension = ".actiff"
    extension_entries = known_extensions.setdefault(extension, [])

    if not any(
        plugin_name in entry.priority
        for entry in extension_entries
    ):
        extension_entries.insert(
            0,
            FileExtension(
                extension=extension,
                priority=[plugin_name],
                name="AIBS Axonal Connectomics TIFF",
                volume_support=True,
            ),
        )


register_actiff()
