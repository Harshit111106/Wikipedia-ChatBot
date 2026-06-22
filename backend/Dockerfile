# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker layer caching
COPY ./requirements.txt /code/requirements.txt

# Install all project dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user with UID 1000 (Mandatory for Hugging Face security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/tmp/.cache

# Copy the rest of the application code, assigning ownership to our user
COPY --chown=user . /code

# Expose the mandatory Hugging Face traffic port
EXPOSE 7860

# Run FastAPI via Uvicorn bound to port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]