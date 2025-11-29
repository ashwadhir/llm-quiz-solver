FROM python:3.10-slim

WORKDIR /app

# 1. Install base system tools (Root)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python libraries including Playwright (Root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Install Playwright SYSTEM dependencies (Must be Root)
# This installs the Ubuntu libraries needed to run Chromium
RUN playwright install-deps

# 4. Create the User
RUN useradd -m -u 1000 user

# 5. Switch to User
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 6. Install the Browser Binary (As User)
# This downloads Chromium into /home/user/.cache/ms-playwright/
# This matches EXACTLY where your error said it was looking.
RUN playwright install chromium

# 7. Copy the rest of the code
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]