FROM gcr.io/kaggle-images/python:latest

#RUN apt update && apt upgrade -y
#RUN apt install curl -y

RUN mkdir /home/rux_ai_s3
COPY . /home/rux_ai_s3
WORKDIR /home/rux_ai_s3

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
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
# Activate venv and run test cases
#RUN . .venv/bin/activate
ENV PATH=./venv/bin:$PATH
RUN bash ./generate_full_game_test_cases.sh


#RUN useradd -ms /bin/bash rusty
#RUN usermod -aG sudo rusty
#
## TODO figure out why rusty can't sudo
#USER rusty
#RUN ./install-rye.sh
#RUN rustup component add rustfmt clippy
#RUN ["/bin/bash", "-c", "bash /home/rusty/.rye/env"]
#WORKDIR /home/rusty/
#
#RUN echo "source /home/rusty/.rye/env " >> ~/.bashrc
#RUN echo "rye install maturin" >> ~/.bashrc
#RUN echo "echo 'No worries about the error above - everything is fine'" >> ~/.bashrc
