# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-dev\
    git \
    && rm -rf /var/lib/apt/lists/*
ADD .ssh/ /root/.ssh/
# Clone the Gestalt Reasoning Benchmark repository
RUN git clone git@github.com:akweury/grb.git /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN pip install opencv-python==4.8.0.74
# Set the default command for training (adjust as needed)
#CMD ["python", "scripts/main.py"]

