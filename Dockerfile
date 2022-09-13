FROM python:3.9-slim-buster
 
COPY ./requirements.txt ./home
COPY ./src ./home
COPY ./output ./home

RUN apt-get update && apt-get -y install procps

RUN pip install --upgrade pip
RUN pip install -r ./home/requirements.txt

WORKDIR /home
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "api:app"]