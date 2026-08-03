FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml requirements.txt ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY . .

# Install papermind in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV PAPERMIND_LLM_PROVIDER=offline

# Command to run Streamlit app by default
CMD ["streamlit", "run", "src/papermind/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
