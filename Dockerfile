FROM python:3.9-slim-buster
 
COPY ./requirements.txt ./home
COPY ./src ./home/src
COPY ./output ./home/output

RUN apt-get update && apt-get -y install procps

RUN pip install --upgrade pip
RUN pip install -r ./home/requirements.txt

WORKDIR /home/src
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "-k", "uvicorn.workers.UvicornWorker", "apiv1:app"]