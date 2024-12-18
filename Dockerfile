# FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1
# WORKDIR /app
# COPY . .
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# ENV DDE_BACKEND="pytorch"
# CMD ["bash"]

ARG MATLAB_RELEASE=R2023b

# Build MATLAB image
FROM mathworks/matlab:$MATLAB_RELEASE

# Declare global
ARG MATLAB_RELEASE

# Install MATLAB Engine API for Python
RUN /bin/sh -c 'cd /opt/matlab/$MATLAB_RELEASE/extern/engines/python && sudo python setup.py install'