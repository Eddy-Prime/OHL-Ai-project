FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Upgrade pip and install standard build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 5000 for Flask
EXPOSE 5800

# Run the Flask app
CMD ["python", "src/interactive_interface/app.py"]
