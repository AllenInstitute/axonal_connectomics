# axonal_connectomics
Repository for tools developed for axonal connectomics

### Level of Support
We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests. Please make pull requests against the develop branch, as we will test changes there before merging into main.

# Stitching modules
## Deskew and zarr conversion
acpreprocessing.stitching_modules.convert_to_n5.tiff_to_ngff
- Sequentially reads image arrays from a tiff stack series for pixel-wise deskew (optional), for computing a downsampling pyramid to a user-defined depth, and for writing out the data volume into a next-generation file format (zarr v3).

## Tile stitching
acpreprocessing.stitching_modules.acstitch.stitch
- Generate point correspondences between tiles from template matching or SIFT features at user-defined resolution level (mip).

## Stitching requirements:
Set these env variables:
```
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export SPARK_LOCAL_IP=127.0.1.1
```
- To run:
```
docker build -t ac-stitch .

docker run \
-v /ACdata:/ACdata \
-v /ispim2_data:/ispim2_data \
-dit ac-stitch

docker exec <container_id> python examples/runmodules.py
```
