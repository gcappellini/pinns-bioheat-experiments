FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1

# Create and set the working directory
RUN mkdir -p /working_dir
WORKDIR /working_dir

COPY . /working_dir

RUN yarn install --production

# Set environment variable for DeepXDE backend
ENV DDE_BACKEND="pytorch"

# Default command to keep the container running
CMD ["bash"]
