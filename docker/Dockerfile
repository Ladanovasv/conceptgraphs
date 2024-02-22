FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && \
    apt-get install -y --no-install-recommends git \
                       wget \
                       nano && \
    rm -rf /var/lib/apt/lists/*

ARG USER=docker_user
ARG UID=1021
ARG GID=1021
# default password
ARG PW=docker_user

# Add user and his password
RUN useradd -m ${USER} --uid=${UID} \
    && echo "${USER}:${PW}" | chpasswd \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER}

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=${CONDA_DIR}/bin:$PATH

RUN conda create -n concept_graphs python=3.10
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda run -n concept_graphs python -m pip --no-cache-dir install tyro open_clip_torch wandb h5py openai hydra-core
RUN conda install -y -n concept_graphs -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

WORKDIR /home/${USER}
USER ${USER}

RUN git clone https://github.com/krrish94/chamferdist.git && cd chamferdist && sed -i 's/c++14/c++17/' setup.py &&\
    conda run -n concept_graphs python -m pip install .

RUN git clone https://github.com/gradslam/gradslam.git 
WORKDIR /home/${USER}/gradslam
RUN git checkout conceptfusion && \
    conda run -n concept_graphs python -m pip install .

WORKDIR /home/${USER}
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything 
WORKDIR /home/${USER}/Grounded-Segment-Anything/segment_anything
RUN conda run -n concept_graphs python -m pip install --no-cache-dir -e .
WORKDIR /home/${USER}/Grounded-Segment-Anything/GroundingDINO
RUN conda run -n concept_graphs python -m pip install --no-cache-dir -e .

WORKDIR /home/${USER}
RUN git clone https://github.com/haotian-liu/LLaVA.git 
WORKDIR /home/${USER}/LLaVA
RUN conda run -n concept_graphs python -m pip install -e .

WORKDIR /home/${USER}/Grounded-Segment-Anything/segment_anything
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*


RUN conda run -n concept_graphs python -m pip install supervision
RUN conda install -y -n concept_graphs pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

USER root
RUN chown -R ${UID}:${GID} /opt/conda
USER ${USER}
WORKDIR /home/${USER}/concept-graphs