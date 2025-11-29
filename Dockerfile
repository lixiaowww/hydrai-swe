# Start from a Python 3.9 base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# Copy the requirements file and install dependencies
COPY requirements_lite.txt .
RUN pip install --no-cache-dir -r requirements_lite.txt

# Copy the source code into the container
COPY ./src /app/src
COPY ./templates /app/templates
COPY ./swe_data.db /app/swe_data.db
COPY ./data /app/data

# Expose the port for the API
EXPOSE 8080

# Set PYTHONPATH to include src directories
ENV PYTHONPATH="${PYTHONPATH}:/app/src:/app/src/api:/app/src/models"

# Command to run the application
# Use gunicorn for production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8080", "src.api.main:app"]
