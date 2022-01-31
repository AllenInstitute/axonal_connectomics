# axonal_connectomics
Repository for tools developed for axonal connectomics

##Stitching requirements:
Set these env variables:
```
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export SPARK_LOCAL_IP=127.0.1.1
```
- to run docker container:
```
sudo docker run \
-v /ACdata:/ACdata \
-v /ispim2_data:/ispim2_data \
-v <local path to axonal connectomics repo>:/axonal_connectomics \
-it ac-stitch
```
- once in the container:
```
cd aaxonal_connectomics
python examples/runmodules.py
```