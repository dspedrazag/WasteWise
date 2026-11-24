# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/wastewise/application

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Compose supplies only ../src as this named context, never the dataset.
COPY --from=project_src data_preparation.py experiment_data.py experiment_training.py experiment_results.py /opt/wastewise/src/
COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
