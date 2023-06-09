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

COPY . $APP_HOME

COPY --from=builder /usr/src/app/wheels /wheels
# https://stackoverflow.com/questions/51115856/docker-failed-to-export-image-failed-to-create-image-failed-to-get-layer
RUN true
COPY --from=builder /usr/src/app/requirements.txt .
RUN true
COPY --from=builder /usr/src/app/proto ./src/proto

RUN pip install --no-cache /wheels/*
RUN pip install dvc[s3]

# generate proto buffer files
RUN python -m grpc_tools.protoc -I ./src/proto/ --python_out=./src/proto/ --grpc_python_out=./src/proto/ ./src/proto/recommendation.proto

# COPY . $APP_HOME
# RUN cp -r proto/* src/proto/

# EXPOSE 50052
# ENTRYPOINT ["python", "./src/serve_grpc.py"]
RUN chown 777 /home/app/web/entrypoint.sh
RUN chmod +x /home/app/web/entrypoint.sh

ENTRYPOINT [ "/home/app/web/entrypoint.sh" ]





