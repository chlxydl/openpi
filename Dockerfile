FROM singularitybase.azurecr.io/base/job/pytorch/acpt-torch2.5.0-py3.10-cuda12.4-ubuntu22.04:20250227T132634623

RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN apt-get update
RUN apt-get install libfuse3-dev fuse3 -y
RUN apt-get install blobfuse2 -y
RUN apt-get install -y cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev


WORKDIR /work