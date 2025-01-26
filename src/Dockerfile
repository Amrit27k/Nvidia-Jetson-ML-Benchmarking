FROM dustynv/l4t-pytorch:r32.7.1

COPY model_cpu_run.py /
COPY requirements.txt /

RUN pip install -r requirements.txt

CMD python3 /model_cpu_run.py 