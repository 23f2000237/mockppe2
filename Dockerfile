# Use lightweight Python image
FROM python:3.11-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose FastAPI port
EXPOSE 8200
# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200"]
