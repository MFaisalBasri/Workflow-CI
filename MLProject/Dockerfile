FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.8.0 \
    pandas==2.3.3 \
    numpy==2.4.6 \
    scipy==1.17.1 \
    cloudpickle==3.1.2 \
    pyarrow==18.1.0

COPY deploy/model /model

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]