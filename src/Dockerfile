FROM dustynv/l4t-pytorch:r36.4.0

RUN apt-get update && apt-get install -y

WORKDIR /app

COPY model_cpu_run.py .
COPY requirements.txt .
COPY images images/
COPY imagenet_classes.txt .
RUN pip3 install -r requirements.txt

# Environment variables
ENV RABBITMQ_HOST=host.docker.internal
ENV DEVICE_TYPE=jetson_orin
ENV DEVICE_ID=orin_1

CMD python3 /model_cpu_run.py 