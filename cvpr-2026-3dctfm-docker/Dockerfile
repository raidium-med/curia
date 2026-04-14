FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 tzdata \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chown user:user /workspace/inputs /workspace/outputs

USER user

ENV PATH="/home/user/.local/bin:${PATH}"

# Upgrade pip and install dependencies
RUN python -m pip install --user -U pip

COPY --chown=user:user requirements.txt /opt/app/requirements.txt
RUN pip install --user -r /opt/app/requirements.txt

# Copy application code
COPY --chown=user:user extract_feat_LP.py extract_feat_LP.sh /opt/app/

# Copy pre-downloaded Curia model weights
COPY --chown=user:user model/ /opt/app/model/

WORKDIR /opt/app/

ENTRYPOINT []
