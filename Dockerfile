# For GPU, use: tensorflow/tensorflow:1.12.0-gpu instead
ARG base_image=tensorflow/tensorflow:1.12.0
FROM $base_image

# install apt-get packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    htop \
    libyaml-0-2 \
    python-gdcm \
    python-opencv \
    pkg-config \
    software-properties-common \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python2 /usr/bin/python

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt \
    && rm -r ~/.cache/pip

ENV TERM=xterm
WORKDIR /imitation
ENV PYTHONPATH=/imitation

CMD ["/bin/bash"]
