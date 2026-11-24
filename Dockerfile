FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pillow numpy
COPY order.py ./
CMD ["python", "order.py", "/frames"]
