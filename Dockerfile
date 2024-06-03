FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
# gcc \
# g++    
# libopenmpi-dev \
# openmpi-bin \
# openmpi-common \
# libopenmpi-dev \
libpython3-dev \
python3-setuptools \
&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install ipython \
scipy \
scikit-learn \
matplotlib \
scikit-optimize \
seaborn \
jupyter \
jupyterlab \
notebook \
pandas \
# mpi4py \
wandb \
deepxde

ENV DDE_BACKEND="pytorch" 
ENV WANDB_API_KEY="3db214c321f91415ac495fd2ac05d678866fb48c"

RUN mkdir -p /working_dir

WORKDIR /working_dir

CMD while true; do sleep 1; done

