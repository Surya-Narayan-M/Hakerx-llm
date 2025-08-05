# --- Stage 1: The Builder ---
# This stage has all the tools needed to install heavy packages.
FROM python:3.11-slim as builder

# Set environment variables to prevent interactive prompts during build
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install only the pip dependencies first. This layer is cached and will only
# re-run if requirements.txt changes, which is a huge speed-up.
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: The Final Application ---
# This stage is a clean, lightweight Python environment.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the pre-compiled wheels from the builder stage. This is much faster
# than running "pip install" again.
COPY --from=builder /app/wheels /wheels

# Install the wheels
RUN pip install --no-cache /wheels/*

# Copy the application code
COPY api_app.py .

# Expose the port that Railway will use. The $PORT variable will be provided by Railway.
# We'll use 8000 as a default if $PORT is not set.
ENV PORT 8000
EXPOSE 8000

# The command to run the application. This replaces the Procfile.
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]


