FROM python:3.10

ARG PROJECT_NAME

ENV CHESS_ENGINE_EXECUTABLE="/usr/games/stockfish"

# Create workdir and copy dependency files
RUN mkdir -p /workdir
COPY . /workdir

# Change shell to be able to easily activate virtualenv
SHELL ["/bin/bash", "-c"]
WORKDIR /workdir

# Install project
RUN apt-get update --fix-missing -qy &&\
    apt-get update -qy &&\
    apt-get install -y apt-utils gosu make ca-certificates sudo git curl tree texlive texlive-latex-extra texlive-fonts-recommended dvipng &&\
    apt-get install -y stockfish polyglot xboard &&\
    chmod 777 "$CHESS_ENGINE_EXECUTABLE"
RUN pip install --upgrade pip virtualenv &&\
    virtualenv /venv &&\
    source /venv/bin/activate &&\
    make install &&\
    chmod -R 777 /venv

# TensorBoard
EXPOSE 6006
# Jupyter Notebook
EXPOSE 8888
# Lightning
EXPOSE 7501
# Lightning
EXPOSE 7502

# Set entrypoint and default container command
ENTRYPOINT ["/workdir/scripts/entrypoint.sh"]
