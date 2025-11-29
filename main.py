import os
import json
import time
import logging
import requests
import pandas as pd
import tabula
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MY_SECRET = os.getenv("tds2_secret")
MY_EMAIL = os.getenv("22f2000771@ds.study.iitm.ac.in")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

def get_page_content(url):
    """Scrape dynamic HTML with Playwright."""
    logger.info(f"Scraping page: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logger.error(f"Playwright failed: {e}")
        return ""


def download_file(file_url):
    """Download file to /tmp safely."""
    try:
        filename = file_url.split("/")[-1].split("?")[0] or "downloaded_file"
        local_path = f"/tmp/{filename}"

        logger.info(f"Downloading {file_url} → {local_path}")

        r = requests.get(file_url, stream=True, timeout=20)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def ask_gemini(prompt, content=""):
    """Call Gemini 2.5 Flash with JSON cleaning."""
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    model = genai.GenerativeModel("gemini-2.5-flash", safety_settings=safety_settings)

    system_inst = (
        "You are a precise data extraction engine. "
        "You NEVER explain. You NEVER write code. "
        "Output ONLY the final value requested."
    )

    full_prompt = f"{system_inst}\n\nCONTEXT:\n{content}\n\nTASK:\n{prompt}"

    try:
        resp = model.generate_content(full_prompt)
        return resp.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return ""


def parse_quiz_page(html):
    """Extract JSON {question, data_url, submit_url} from HTML."""
    prompt = """
    Extract quiz instructions from the HTML.
    Return ONLY valid JSON:
    {
        "question": "...",
        "data_url": "... or null",
        "submit_url": "..."
    }
    """

    raw = ask_gemini(prompt, html[:40000])
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except:
        logger.error("Gemini JSON parse failed; using fallback.")
        return {"question": "Extract answer from page.", "data_url": None, "submit_url": None}


# --------------------------------------------------------------------
# CORE SOLVER
# --------------------------------------------------------------------

def solve_quiz_loop(start_url):
    current_url = start_url
    deadline = time.time() + 170  # 3 minutes buffer

    while current_url and time.time() < deadline:

        logger.info(f"Processing quiz: {current_url}")

        html = get_page_content(current_url)
        task = parse_quiz_page(html)

        # absolute URLs
        if task.get("data_url"):
            task["data_url"] = urljoin(current_url, task["data_url"])
        if task.get("submit_url"):
            task["submit_url"] = urljoin(current_url, task["submit_url"])

        question = task["question"]
        data_url = task["data_url"]
        submit_url = task["submit_url"]

        logger.info(f"Parsed task: {task}")

        # ----------------------------------------------------------------
        # SOLVE QUIZ (2 attempts allowed)
        # ----------------------------------------------------------------
        answer = None

        for attempt in range(2):

            if data_url:
                file_path = download_file(data_url)

                if not file_path:
                    answer = ""
                    continue

                # CSV CASE
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    preview = df.head(30).to_string()
                    prompt = f"Using this CSV data:\n{preview}\n\nQuestion: {question}\nONLY output the answer."
                    answer = ask_gemini(prompt)

                # PDF CASE
                elif file_path.endswith(".pdf"):
                    try:
                        dfs = tabula.read_pdf(file_path, pages="all")
                        if dfs:
                            table = dfs[0].to_string()
                            prompt = f"PDF table:\n{table}\n\nQuestion: {question}\nONLY output answer."
                            answer = ask_gemini(prompt)
                        else:
                            answer = ""
                    except:
                        answer = ""

                # GENERIC FILE
                else:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(10000)
                    except:
                        content = ""

                    extraction_prompt = (
                        f"QUESTION: {question}\n"
                        f"CONTENT:\n{content}\n"
                        f"ONLY OUTPUT the required answer."
                    )
                    answer = ask_gemini(extraction_prompt)

            else:
                # NO FILE → answer directly from HTML
                extraction_prompt = (
                    f"HTML:\n{html[:15000]}\n\n"
                    f"Question: {question}\n"
                    f"Extract ONLY the answer."
                )
                answer = ask_gemini(extraction_prompt)

            if answer:
                break

        logger.info(f"Computed Answer: {answer}")

        # ----------------------------------------------------------------
        # SUBMIT ANSWER
        # ----------------------------------------------------------------
        try:
            payload = {
                "email": MY_EMAIL,
                "secret": MY_SECRET,
                "url": current_url,
                "answer": answer
            }

            r = requests.post(submit_url, json=payload, timeout=20)
            r.raise_for_status()
            resp = r.json()

            logger.info(f"Submission Response: {resp}")

            if resp.get("correct"):
                current_url = resp.get("url")  # next quiz
            else:
                # incorrect → try resubmit or move to next if provided
                current_url = resp.get("url")
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return {"error": "submission failed"}

    return {"status": "completed"}



# --------------------------------------------------------------------
# API ENDPOINT
# --------------------------------------------------------------------

@app.post("/")
def handler(req: QuizRequest):
    if req.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        result = solve_quiz_loop(req.url)
        return {"ok": True, "result": result}
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise HTTPException(status_code=500, detail="Server error")