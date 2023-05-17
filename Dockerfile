From python:3.6 as builder

workdir usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install gcc -y \
    && apt-get install git -y \
    && apt-get clean

Run pip install --upgrade pip

COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# download the proto files
RUN git clone https://github.com/atharvatechnology/ronb-grpc-proto.git

# copy the proto files to proto dir
RUN cp -r ronb-grpc-proto/proto ./proto

# rm the cloned repo
RUN rm -rf ronb-grpc-proto

COPY . .

# production environment
FROM python:3.6-slim

RUN mkdir -p /home/app

# RUN addgroup app && adduser app

ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install gcc -y \
    && apt-get clean

COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .

RUN pip install --no-cache /wheels/*

# generate proto buffer files
RUN python -m grpc_tools.protoc -I ./proto/ --python_out=./proto/ --grpc_python_out=./proto/ ./proto/recommendation.proto

COPY . $APP_HOME

EXPOSE 50052
ENTRYPOINT ["python", "src/grpc_server.py"]






