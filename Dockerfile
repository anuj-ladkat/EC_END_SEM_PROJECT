# Use NVIDIA Jetson base image
FROM nvcr.io/nvidia/l4t-base:r35.2.1

# Install basic dependencies and tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    libssl-dev \
    libffi-dev \
    libatlas-base-dev \
    libjasper-dev \
    libgtk2.0-dev \
    libopencv-dev \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    libqtgui4 \
    libqt4-test \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean

# Install dlib and face_recognition dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir dlib

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install Python packages
RUN pip3 install --no-cache-dir \
    flask \
    numpy \
    opencv-python-headless \
    face_recognition \
    requests \
    smtplib

# Expose the application port
EXPOSE 5003

# Start the Flask application
CMD ["python3", "app.py"]
