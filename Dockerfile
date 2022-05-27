FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y git && \
    python -m pip install --no-cache-dir --upgrade pip

RUN git clone --branch=main https://github.com/google-research/t5x && \
    cd ./t5x && \
    python -m pip install --no-cache-dir -e . && \
    cd ..

RUN git clone -b new_model/LongT5 https://github.com/stancld/transformers.git && \
    cd ./transformers && \
    python -m pip install --no-cache-dir -e . && \
    cd ..

WORKDIR /workspace

RUN git clone https://github.com/stancld/longt5-eval.git  && \
    cd longt5-eval && \
    git clone https://github.com/google/flaxformer.git && \
    mv ./flaxformer/flaxformer/ _flaxformer/ && \
    rm -rf flaxformer/ && \
    mv _flaxformer/ flaxformer/ && \
    python -m pip install --no-cache-dir -r flaxformer_requirements.txt

WORKDIR /workspace/longt5-eval

COPY google-checkpoints google-checkpoints
