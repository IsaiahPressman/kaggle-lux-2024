# TODO: Try using gcr.io/kaggle-images/python:v155
FROM gcr.io/kaggle-images/python:v154

#RUN apt update && apt upgrade -y
#RUN apt install curl -y

RUN mkdir /home/rux_ai_s3
COPY . /home/rux_ai_s3
# This directory is not copied by docker
RUN mkdir /home/rux_ai_s3/train_outputs
WORKDIR /home/rux_ai_s3

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=/root/.cargo/bin:$PATH
# Update rust nightly
RUN rustup update nightly
RUN rustup component add rustfmt --toolchain nightly
# Install rye
ENV PATH=/root/.rye/shims:$PATH
RUN curl -sSf https://rye.astral.sh/get | RYE_VERSION="0.41.0" RYE_INSTALL_OPTION="--yes" bash
# Install maturin
RUN rye install maturin
# Install packages
RUN rye sync
# Activate venv and run make prepare
ENV PATH=/home/rux_ai_s3/.venv/bin/:$PATH
RUN make prepare-agent
# Tar compiled submission for export
RUN tar --exclude="*__pycache__*" --transform "s,^python/,," -czvf submission.tar.gz python/main.py python/rux_ai_s3
