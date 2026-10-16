# Use an official lightweight Python runtime environment image
FROM python:3.11-slim

# Set system environment variables to optimize Python performance inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish application deployment context workspace directory
WORKDIR /app

# Copy system software dependencies manifest file first to leverage Docker layer caching
COPY requirements.txt .

# Install package dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy remaining codebase assets and serialised binary pickles into the image layer
COPY config/ ./config/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY main.py .

# Expose the app context execution entrypoint
CMD ["python", "main.py"]
