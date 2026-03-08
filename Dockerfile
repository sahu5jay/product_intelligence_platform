FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r deployment/requirements.txt

# Expose API port
EXPOSE 5000

# Run Flask API
CMD ["python", "deployment/api/main.py"]