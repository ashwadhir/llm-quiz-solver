import os
import json
import requests
import logging
import pandas as pd
import numpy as np
import tabula
import google.generativeai as genai
import re
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MY_SECRET = os.getenv("MY_SECRET", "tds2_secret")
MY_EMAIL = os.getenv("MY_EMAIL", "22f2000771@ds.study.iitm.ac.in")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- HELPERS ---

def get_page_content(url, wait_ms: int = 3000):
    """Scrapes dynamic HTML using Playwright and returns rendered HTML (string)."""
    logger.info(f"Scraping (render): {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_timeout(wait_ms)
        content = page.content()
        browser.close()
        return content


def download_file(file_url):
    """Downloads a file to /tmp/ and returns local path. Returns None on failure."""
    try:
        parsed = urlparse(file_url)
        raw_name = os.path.basename(parsed.path) or "downloaded_data"
        filename = raw_name.split("?")[0]
        local_filename = f"/tmp/{filename}"

        logger.info(f"Downloading {file_url} -> {local_filename}")
        with requests.get(file_url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_filename
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def prepare_local_html(file_path, base_url=None):
    """If the downloaded file is HTML and references scripts, try to download those
    scripts to the same /tmp folder so file:// rendering will execute them.
    base_url is used to resolve relative script URLs.
    Returns the original file_path (modified in-place) or the same path on failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception:
        return file_path

    scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if not scripts:
        return file_path

    for script in scripts:
        try:
            # Resolve script URL
            if script.startswith("http://") or script.startswith("https://"):
                script_url = script
            else:
                # If base_url is provided and looks like an http(s) url, use urljoin
                if base_url and base_url.startswith("http"):
                    script_url = urljoin(base_url, script)
                else:
                    # Best effort: if file_path is /tmp/name, try to fetch from same domain
                    script_url = urljoin(base_url or "", script)

            local_name = os.path.basename(urlparse(script_url).path) or script
            local_path = os.path.join(os.path.dirname(file_path), local_name)

            # If already exists, skip
            if os.path.exists(local_path):
                continue

            resp = requests.get(script_url, timeout=5)
            if resp.status_code == 200:
                with open(local_path, "wb") as sf:
                    sf.write(resp.content)
                logger.info(f"Fetched script {script_url} -> {local_path}")
        except Exception as e:
            logger.debug(f"Could not fetch script {script}: {e}")

    return file_path


def sanitize_content(text: str) -> str:
    if not text:
        return ""
    text = text.replace(MY_EMAIL, " [REDACTED_EMAIL] ")
    # remove IITM roll-like tokens (best-effort)
    text = re.sub(r'22[a-z]\d{6,}', ' [REDACTED_ID] ', text, flags=re.IGNORECASE)
    return text


def ask_gemini(prompt, content=""):
    """Call Gemini model (if configured). If Gemini not configured, return empty string.
    Keep calls minimal: prefer python-native computation where possible.
    """
    if not GEMINI_API_KEY:
        logger.debug("GEMINI_API_KEY not set — skipping Gemini call")
        return ""

    # Safety settings left permissive for this exercise (mirrors original)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
        system_instruction = (
            "You are a precise data extraction engine. Output ONLY the requested value."
            " Do NOT return the user's email or ID."
        )
        full_prompt = f"{system_instruction}\n\nCONTEXT:\n{content}\n\nTASK:\n{prompt}"
        response = model.generate_content(full_prompt)
        if response and getattr(response, 'parts', None):
            return response.text.strip()
        return ""
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return ""


def parse_quiz_page(html_content: str):
    """Extract task JSON from the HTML using a lightweight approach with
    a Gemini attempt as fallback. Returns a dict with question/data_url/submit_url.
    """
    # Small regex heuristics first
    question_text = None
    qmatch = re.search(r"(?:Question:|Q\d+\.|To what|Calculate)([\s\S]{0,300})", html_content, re.IGNORECASE)
    if qmatch:
        question_text = qmatch.group(0).strip()

    prompt = (
        "Analyze this HTML. Extract the JSON task.\n"
        "JSON Format:\n{\n  \"question\": \"The exact question text.\",\n  \"data_url\": \"URL of file to download (or null)\",\n  \"submit_url\": \"URL to POST answer to\",\n}\nReturn ONLY raw JSON."
    )

    cleaned_text = ask_gemini(prompt, html_content[:50000])
    if cleaned_text:
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned_text)
            if not data.get('question') and question_text:
                data['question'] = question_text
            return data
        except Exception:
            logger.debug("Gemini returned non-JSON or parse failed")

    # Fallback simple parsing
    data_url = None
    submit_url = None

    # find links that look like data or submit
    data_match = re.search(r"(?:href|src)=[\"']([^\"']*data[^\"']*)[\"']", html_content, re.IGNORECASE)
    if data_match:
        data_url = data_match.group(1)

    submit_match = re.search(r"POST this JSON to ([^\s'\"]+)", html_content)
    if submit_match:
        submit_url = submit_match.group(1)

    return {
        "question": question_text or "Extract the secret code from the page.",
        "data_url": data_url,
        "submit_url": submit_url,
    }


def clean_html_text(raw_html: str) -> str:
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return ' '.join(cleantext.split())

# --- CORE SOLVER ---

def solve_quiz_loop(start_url):
    current_url = start_url

    while current_url:
        try:
            logger.info(f"--- Processing Level: {current_url} ---")

            # 1. Fetch rendered page content (JS executed)
            try:
                html = get_page_content(current_url, wait_ms=3500)
            except Exception as e:
                logger.error(f"Playwright scrape failed for {current_url}: {e}")
                # Fallback to requests
                try:
                    html = requests.get(current_url, timeout=8).text
                except Exception as e2:
                    logger.error(f"HTTP fallback failed: {e2}")
                    break

            # 2. Parse task
            task = parse_quiz_page(html)

            # Normalize relative URLs
            if task.get("data_url"):
                task["data_url"] = urljoin(current_url, task["data_url"]) if task["data_url"] else None
            if task.get("submit_url"):
                task["submit_url"] = urljoin(current_url, task["submit_url"]) if task["submit_url"] else None

            if not task.get("question"):
                task["question"] = "Extract the main answer or secret code from the page context."

            logger.info(f"Task Parsed: {task}")

            # Attempt loop (try twice with slightly different strategies)
            for attempt in range(2):
                answer = None

                if task.get("data_url"):
                    file_path = download_file(task["data_url"])

                    if not file_path:
                        answer = ""

                    # CSV handling: do reliable numeric summing
                    elif file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
                        try:
                            df = pd.read_csv(file_path)

                            # Convert all cols to numeric where possible
                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

                            # Drop columns that are completely NaN
                            df = df.dropna(axis=1, how='all')

                            total = df.sum(numeric_only=True).sum()
                            # If total is float but represents int, cast safely
                            if pd.isna(total):
                                total = 0
                            if float(total).is_integer():
                                total = int(total)
                            else:
                                total = float(total)

                            logger.info(f"Computed CSV total: {total}")
                            answer = total
                        except Exception as e:
                            logger.error(f"CSV processing failed: {e}")
                            answer = 0

                    # PDF handling: use tabula, then numeric aggregation if possible
                    elif file_path.endswith('.pdf'):
                        try:
                            dfs = tabula.read_pdf(file_path, pages='all')
                            if dfs and len(dfs) > 0:
                                # Flatten to first table and attempt numeric sum
                                tbl = dfs[0]
                                for col in tbl.columns:
                                    tbl[col] = pd.to_numeric(tbl[col].astype(str).str.replace(',', ''), errors='coerce')
                                tbl = tbl.dropna(axis=1, how='all')
                                total = tbl.sum(numeric_only=True).sum()
                                if pd.isna(total):
                                    answer = 0
                                else:
                                    answer = int(total) if float(total).is_integer() else float(total)
                            else:
                                answer = ""
                        except Exception as e:
                            logger.error(f"PDF parse error: {e}")
                            answer = ""

                    else:
                        # Generic downloaded file (likely HTML that runs JS)
                        try:
                            # sanitize raw file
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                raw = f.read()
                        except Exception:
                            raw = ""

                        raw = sanitize_content(raw)

                        # Try to download any referenced scripts so file:// rendering works
                        try:
                            prepare_local_html(file_path, base_url=task.get('data_url'))
                        except Exception as e:
                            logger.debug(f"prepare_local_html failed: {e}")

                        # Render the downloaded file with Playwright so scripts execute
                        try:
                            rendered = get_page_content('file://' + file_path, wait_ms=1500)
                            cleaned = clean_html_text(rendered)
                            cleaned = sanitize_content(cleaned)

                            # Prefer extraction via simple heuristics: look for large alphanumeric tokens
                            token_match = re.search(r"[A-Za-z0-9]{6,}", cleaned)
                            if token_match:
                                answer = token_match.group(0)
                            else:
                                # As a last resort, ask Gemini (if configured)
                                extraction_prompt = (
                                    f"QUESTION: {task['question']}\nCONTENT: {cleaned}\nTASK: Extract the secret code. Return ONLY the code."
                                )
                                answer = ask_gemini(extraction_prompt, content=cleaned[:30000])

                            if not answer:
                                # fallback to cleaned raw
                                answer = cleaned.strip()

                        except Exception as e:
                            logger.error(f"Rendered local file failed: {e}")
                            # fallback heuristics on raw content
                            token_match = re.search(r"[A-Za-z0-9]{6,}", raw)
                            answer = token_match.group(0) if token_match else raw.strip()

                else:
                    # No data_url: try to extract directly from rendered HTML
                    cleaned_html = sanitize_content(html)
                    # Heuristic extraction first
                    token_match = re.search(r"secret\W*[:=]?\W*([A-Za-z0-9-]{4,})", cleaned_html, re.IGNORECASE)
                    if token_match:
                        answer = token_match.group(1)
                    else:
                        # Use Gemini as fallback
                        tone = "Think step by step." if attempt > 0 else "Answer directly."
                        answer = ask_gemini(f"Question: {task['question']}\n{tone}", content=cleaned_html[:30000])

                # Normalize answer type and strip sensitive tokens
                try:
                    ans_str = str(answer).strip()
                    ans_str = ans_str.replace('"', '').replace("'", '')
                    ans_str = ans_str.replace(MY_EMAIL, '').replace('22f2000771', '')
                    # If it's numeric-looking, convert
                    if re.fullmatch(r"-?\d+", ans_str):
                        answer = int(ans_str)
                    elif re.fullmatch(r"-?\d+\.\d+", ans_str):
                        answer = float(ans_str)
                    else:
                        answer = ans_str
                except Exception:
                    pass

                # Prepare submission
                payload = {
                    "email": MY_EMAIL,
                    "secret": MY_SECRET,
                    "url": current_url,
                    "answer": answer
                }

                if not task.get("submit_url"):
                    task["submit_url"] = urljoin(current_url, "/submit")

                logger.info(f"Submitting: {payload}")
                try:
                    res = requests.post(task["submit_url"], json=payload, timeout=15)
                    try:
                        res_json = res.json()
                    except Exception:
                        res_json = {}
                except Exception as e:
                    logger.error(f"Submit failed: {e}")
                    res_json = {}

                logger.info(f"Result: {res_json}")

                if res_json.get("correct"):
                    current_url = res_json.get("url")
                    break
                else:
                    logger.warning("Answer incorrect.")
                    if attempt == 0:
                        logger.info("Retrying with alternative strategy...")
                        continue
                    else:
                        current_url = res_json.get("url")
                        break

            if not current_url:
                logger.info("No next URL provided. Quiz Finished.")
                break

        except Exception as e:
            logger.error(f"CRITICAL ERROR: {e}")
            break


@app.post("/")
async def receive_task(task: QuizRequest, background_tasks: BackgroundTasks):
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(solve_quiz_loop, task.url)
    return {"message": "Processing"}