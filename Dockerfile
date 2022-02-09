# syntax=docker/dockerfile:1

FROM openjdk:8
# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN apt-get update -y

RUN apt-get install -y python


WORKDIR /ac-stitch

COPY requirements.txt requirements.txt
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3.6 python3-pip
RUN apt install -y python-is-python3

RUN pip3 install -r requirements.txt

RUN apt-get install -y curl 
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output miniconda.sh
RUN bash miniconda.sh -b
ENV PATH="/root/miniconda3/bin:$PATH"

RUN conda install -c conda-forge z5py

RUN pip install argschema
RUN pip install imageio
RUN pip install natsort
RUN pip install scikit-image

COPY . .

RUN pip install .


