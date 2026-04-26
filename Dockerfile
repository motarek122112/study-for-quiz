FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN echo "Preparing container for Run ID: ${RUN_ID}"

CMD ["python", "-c", "print('Model container is running')"]
