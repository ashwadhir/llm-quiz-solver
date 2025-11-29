FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies + Java (Required for Tabula)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 2. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 3. SET PLAYWRIGHT LOCATION (The Critical Fix)
# We force Playwright to install browsers in a global folder, not /root/
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN mkdir -p $PLAYWRIGHT_BROWSERS_PATH

# 4. Install Playwright Browsers & Dependencies
RUN playwright install-deps
RUN playwright install chromium

# 5. Create a non-root user
RUN useradd -m -u 1000 user

# 6. Grant permissions to the user
# Give user access to the app folder AND the browser folder
RUN chown -R user:user /app
RUN chown -R user:user $PLAYWRIGHT_BROWSERS_PATH

# 7. Copy application code (with user ownership)
COPY --chown=user . .

# 8. Switch to user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]