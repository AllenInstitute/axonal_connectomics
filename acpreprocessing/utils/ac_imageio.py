import imageio
import imageio.plugins.tifffile


class ACTiffFormat(imageio.plugins.tifffile.TiffFormat):
    class Reader(imageio.plugins.tifffile.TiffFormat.Reader):
        def _get_meta_data(self, index):
            meta = super()._get_meta_data(index)
            page = self._tf.pages[index or 0]
            meta["acquisition_md"] = page.tags[51123].value
            return meta

        def _get_data(self, index):
            im, meta = super()._get_data(index)
            return im, self._get_meta_data(index or 0)


fmt = ACTiffFormat(
    "actiff",
    "tiff format for AIBS axonal connectomics",
    ".actiff",
    "iIvV"
)

imageio.formats.add_format(fmt, overwrite=True)

