FROM continuumio/miniconda3:23.10.0-1 as acpreprocessing

# Update conda and install Python
RUN conda update -y conda && \
    conda install -y python=3.10 && \
    conda clean -a

# Set bash as the default shell
SHELL ["/bin/bash", "-c"]

# Install curl
RUN apt-get update && apt-get install -y curl

# Copy your project code into the container
COPY . /ax_conn

# Set working directory
WORKDIR /ax_conn

# Install dependencies and run migrations
RUN conda install -y pip && \
    conda install -y -c conda-forge gcc && \
    curl -fsSL https://pixi.sh/install.sh | bash 

ENV PATH="/root/.pixi/bin:$PATH"

RUN pixi install && \
    conda clean -a

# Final working directory (adjust as needed)
WORKDIR /ax_conn
