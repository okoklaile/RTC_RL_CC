FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Vancouver

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo "$TZ" > /etc/timezone && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    python3-setuptools \
    wget \
    unzip \
    mahimahi \
    jq \
    tzdata \
    ffmpeg \
    gocr \
    imagemagick \
    iputils-ping \
    iperf3 \
    iproute2 \
    net-tools \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip3 install pytest numpy requests soundfile gym protobuf==3.6.1
RUN pip3 install opencv-python scikit-image
#RUN pip3 install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
#--index-url https://download.pytorch.org/whl/cpu

# 假设你已经将文件下载到 dockers/downloads/ 目录下
COPY downloads/ubuntu-18.04-vmaf.zip /tmp/

# 使用本地文件进行解压和安装
RUN cd /tmp && \
    unzip -o ubuntu-18.04-vmaf.zip && \
    chmod 755 vmaf && \
    mv vmaf /usr/bin && \
    rm ubuntu-18.04-vmaf.zip

WORKDIR /app
COPY alphartc/target/ /app/alphartc/target/
COPY . /app/
RUN useradd -m user && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R user:user /app && chmod -R 755 /app
# USER user

CMD [ "/bin/bash" ]
