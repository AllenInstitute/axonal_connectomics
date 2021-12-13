import json

import imageio
import imageio.plugins.tifffile


class ACTiffFormat(imageio.plugins.tifffile.TiffFormat):
    class Reader(imageio.plugins.tifffile.TiffFormat.Reader):
        _acquisition_md_tags = [
            51123,
            "MicroManagerMetadata"
            ]

        def _get_meta_data(self, index):
            meta = super()._get_meta_data(index)
            page = self._tf.pages[index or 0]
            for tag_key in self._acquisition_md_tags:
                try:
                    meta["MicroManagerMetadata"] = page.tags[tag_key].value
                    break
                except KeyError:
                    pass
            return meta

        def _get_data(self, index):
            im, meta = super()._get_data(index)
            return im, self._get_meta_data(index or 0)

    class Writer(imageio.plugins.tifffile.TiffFormat.Writer):
        _acquisition_md_tags = [
            51123,
            "MicroManagerMetadata"
        ]
        _description_encoding = "latin-1"

        def _append_data(self, im, meta):
            meta = (meta if meta is not None else (
                self._meta if self._frames_written == 0 else {}))
            acquisition_md_tags = [
                ("MicroManagerMetadata", "s", 0,
                 json.dumps(meta[md_tag]), True)
                for md_tag in {*self._acquisition_md_tags} & meta.keys()
            ]
            try:
                meta["extratags"] += acquisition_md_tags
            except KeyError:
                meta["extratags"] = acquisition_md_tags
            try:
                meta["description"] = meta["description"].encode(
                    self._description_encoding)
            except (AttributeError, KeyError):
                pass
            return super()._append_data(im, meta)


fmt = ACTiffFormat(
    "actiff",
    "tiff format for AIBS axonal connectomics",
    ".actiff .tiff .tif",
    "iIvV"
)

imageio.formats.add_format(fmt, overwrite=True)
