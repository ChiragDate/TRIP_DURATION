FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ ./src/
COPY models/ ./models/

# Make port 8000 available to the world outside the container
EXPOSE 8000

# Run the API service
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]