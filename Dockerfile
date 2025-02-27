# Use the official PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime

ADD .ssh/ /root/.ssh/
# Clone the Gestalt Reasoning Benchmark repository
RUN git clone git@github.com:akweury/grb.git /app

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel
# Install Python dependencies with --no-cache-dir
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install opencv-python==4.8.0.74
# Set the default command for training (adjust as needed)
#CMD ["python", "scripts/main.py"]

