FROM ubuntu:22.04

WORKDIR /omniparser

# Copy repo into working directory
COPY repo /omniparser

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    python3 python3-pip

# Install python dependencies
RUN pip3 install -r requirements.txt

# Convert the safetensor to .pt file.
RUN python3 weights/convert_safetensor_to_pt.py

# Run the demo to install OCR models
RUN python3 demo.py
