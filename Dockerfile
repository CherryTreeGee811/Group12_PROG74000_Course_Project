FROM python:3.13-slim-trixie AS builder
WORKDIR /app
ENV GIT_PYTHON_REFRESH=quiet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python features/build_dataset.py

FROM python:3.13-slim-trixie AS final
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./features/technical_indicators.py ./features/
RUN mkdir -p ./models/saved
RUN mkdir -p ./data
RUN mkdir -p ./features
RUN mkdir -p ./app
COPY --from=builder /app/config.yaml .
COPY --from=builder /app/data/fetch_price.py ./data
COPY --from=builder /app/data/cache ./data/cache
COPY --from=builder /app/models/saved ./models/saved
COPY --from=builder /app/models/train.py ./models/
COPY --from=builder /app/models/predict.py ./models/
COPY --from=builder /app/models/evaluate.py ./models/
COPY --from=builder /app/app/main.py ./app
COPY --from=builder /app/app/api.py ./app
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 5000
ENV FLASK_APP=app/api.py
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
