#FROM python:3.10.12
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV HOME=/workspace

# Install
RUN apt-get update
RUN apt-get install -y vim \
                       libgl1-mesa-glx \
                       libglib2.0-0 \
                       wget

# Download weight before
RUN mkdir -p /workspace/.cache/torch/hub/checkpoints/
RUN wget -qO /workspace/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth https://download.pytorch.org/models/resnet50-11ad3fa6.pth
RUN wget -qO /workspace/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


# Copy code
COPY src src
COPY config config


# Run param
#RUN wandb offline
