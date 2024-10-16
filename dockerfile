FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update
RUN apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    unzip wget git curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install pytorch, transformers, torchtext, torchvision
# torchvision breaks after 2.4.0 so we need to install torch<2.4.0
RUN pip install "torch<2.4.0" torchvision torchtext transformers

# Default python version
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Need libstdc++ so install build-essential
RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install g++-11 -y

# Make python3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install openssl
RUN cd /tmp && \
    wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/openssl_1.1.1f-1ubuntu2_amd64.deb && \
    wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl-dev_1.1.1f-1ubuntu2_amd64.deb && \
    wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb \
    libssl-dev_1.1.1f-1ubuntu2_amd64.deb \
    openssl_1.1.1f-1ubuntu2_amd64.deb

# Clone Pencil
RUN cd /root \
    && git clone https://github.com/lightbulb128/Pencil.git

WORKDIR /root/Pencil

# Copy library-files.zip to /tmp, unzip it
COPY library-files.zip /tmp/library-files.zip
RUN mkdir /tmp/library-files && unzip /tmp/library-files.zip -d /tmp/library-files \
    && mkdir -p /root/Pencil/pencil-fullhe/tools \
                /root/Pencil/pencil-prep/tools \
                /root/Pencil/pencil-fullhe/logs \
                /root/Pencil/pencil-prep/logs \
    && cp /tmp/library-files/*.so /root/Pencil/pencil-fullhe/tools/ \
    && cp /tmp/library-files/*.so /root/Pencil/pencil-prep/tools/

# Add the library path to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/root/Pencil/pencil-fullhe/tools:$LD_LIBRARY_PATH