FROM python:3.10-slim

WORKDIR /app

# Install system dependencies + JAVA (Required for Tabula/PDFs)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright
RUN playwright install chromium
RUN playwright install-deps

COPY . .

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]