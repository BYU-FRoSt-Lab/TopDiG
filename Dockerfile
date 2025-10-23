FROM python:3.9

# Install uv
RUN pip install --no-cache-dir uv

# Install system dependencies for GDAL and Detectron
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    gcc \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install GDAL Python bindings (matching system GDAL version)
RUN uv pip install --system --no-cache GDAL==$(gdal-config --version)

# Install Caffe (required for Detectron)
# RUN apt-get update && apt-get install -y \
#     libprotobuf-dev \
#     libleveldb-dev \
#     libsnappy-dev \
#     libopencv-dev \
#     libhdf5-serial-dev \
#     protobuf-compiler \
#     libatlas-base-dev \
#     libboost-all-dev \
#     libgflags-dev \
#     libgoogle-glog-dev \
#     liblmdb-dev \
#     && rm -rf /var/lib/apt/lists/*
#
# # Install Detectron (original version)
# WORKDIR /opt
# RUN git clone https://github.com/facebookresearch/Detectron.git && \
#     cd Detectron && \
#     uv pip install --system --no-cache -r requirements.txt && \
#     cd lib && \
#     make

# Add Detectron to Python path
ENV PYTHONPATH="/opt/Detectron:${PYTHONPATH}"

# Set working directory
WORKDIR /app

# Copy source directory
COPY ./ ./src/

# Copy and install requirements
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Default command
CMD ["python"]
