# axonal_connectomics
Repository for tools developed for axonal connectomics

# Level of support

We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues and pull requests, but you should not expect an active response.


##Stitching requirements:
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
