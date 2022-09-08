FROM python:3.6-slim-buster as base

FROM base as builder 

COPY ./requirements.txt ./scripts/install.sh ./
RUN ./install.sh && python -m venv /opt/venv

# setup venv as path
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

FROM base

RUN apt-get update \
    && apt-get -y install procps

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/apps/project

# Idle
CMD ["tail", "-f", "/dev/null"]