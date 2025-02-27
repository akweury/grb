# Use the official NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ttf-mscorefonts-installer \
    && rm -rf /var/lib/apt/lists/*


RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime


# Ensure SSH key has correct permissions (if using SSH cloning)
ADD .ssh/ /root/.ssh/
RUN chmod 600 /root/.ssh/id_rsa && ssh-keyscan github.com >> /root/.ssh/known_hosts


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

