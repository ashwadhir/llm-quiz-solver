import os
import json
import time
import logging
import re
import requests
import pandas as pd
import tabula
import google.generativeai as genai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

def get_page_content(url):
    logger.info(f"Scraping page: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            page.wait_for_timeout(1500)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logger.error(f"Playwright error: {e}")
        return ""

def download_file(file_url):
    try:
        filename = file_url.split("/")[-1].split("?")[0] or "downloaded"
        local_path = f"/tmp/{filename}"
        logger.info(f"Downloading {file_url} → {local_path}")
        r = requests.get(file_url, stream=True, timeout=20)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def ask_gemini(prompt, content=""):
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        model = genai.GenerativeModel("gemini-2.5-flash", safety_settings=safety_settings)
        system_inst = "You are a precise data extraction engine. Output ONLY the requested result."
        full_prompt = f"{system_inst}\n\nCONTEXT:\n{content}\n\nTASK:\n{prompt}"
        resp = model.generate_content(full_prompt)
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return ""

def clean_llm_output(txt: str):
    """Strip code fences, backticks, markdown and whitespace."""
    if not isinstance(txt, str): return ""
    s = txt.strip()
    # remove triple/backtick fences
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # remove single backticks and surrounding quotes
    s = s.replace("`", "").strip()
    # if wrapped in a JSON block, try to extract the JSON
    try:
        # attempt to find JSON substring
        jmatch = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if jmatch:
            js = jmatch.group(0)
            return js.strip()
    except:
        pass
    return s

def try_parse_json_if_present(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except:
        # try to fix common single quotes / trailing commas
        try:
            s2 = s.replace("'", '"')
            s2 = re.sub(r",\s*}", "}", s2)
            return json.loads(s2)
        except:
            return None

def extract_secret_from_text(text: str):
    """Heuristic: look for short alphanumeric token (6-40 chars) labeled 'secret' or 'code'."""
    # look for labels
    m = re.search(r"(?:secret|code|flag)[:\s]*([A-Za-z0-9_\-]{4,80})", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: any long-ish alnum token
    m2 = re.search(r"([A-Za-z0-9_\-]{6,80})", text)
    if m2:
        return m2.group(1)
    return text.strip()[:200]

# --------------------------------------------------------------------
# PARSE PAGE (using LLM as fallback)
# --------------------------------------------------------------------

def parse_quiz_page(html):
    # quick HTML heuristics
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    # try to find obvious submit URL / data links
    submit = None
    data_link = None
    # find anchor with /submit
    a = soup.find("a", href=True)
    if a and "/submit" in a["href"]:
        submit = urljoin("https://example.com", a["href"])
    # find any links ending with common data files
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if any(ext in href for ext in [".csv", ".pdf", "?"]):
            data_link = href
            break

    # Minimal JSON-like detection in page
    json_like = None
    jmatch = re.search(r"\{[\s\S]{10,2000}\}", html)
    if jmatch:
        json_like = jmatch.group(0)

    # If we found core pieces, return them; else use LLM fallback
    if text and (submit or data_link or json_like):
        return {
            "question": text[:2000],
            "data_url": data_link,
            "submit_url": submit
        }

    # LLM fallback (limited prompt)
    prompt = """
    Analyze this HTML and extract a small JSON with keys:
    { "question": "...", "data_url": "... or null", "submit_url": "... or null" }
    Return ONLY JSON.
    """
    llm_out = ask_gemini(prompt, content=html[:30000])
    cleaned = clean_llm_output(llm_out)
    parsed = try_parse_json_if_present(cleaned)
    if parsed and isinstance(parsed, dict):
        return {
            "question": parsed.get("question"),
            "data_url": parsed.get("data_url"),
            "submit_url": parsed.get("submit_url")
        }
    # fallback generic
    return {"question": text[:2000], "data_url": None, "submit_url": None}

# --------------------------------------------------------------------
# SOLVER LOOP
# --------------------------------------------------------------------

def solve_quiz_loop(start_url):
    current_url = start_url
    deadline = time.time() + 170  # give ~170s to finish
    logger.info("Background solver started.")
    while current_url and time.time() < deadline:
        try:
            logger.info(f"Processing quiz: {current_url}")
            html = get_page_content(current_url)
            task = parse_quiz_page(html)
            # make absolute
            if task.get("data_url"):
                task["data_url"] = urljoin(current_url, task["data_url"])
            if task.get("submit_url"):
                task["submit_url"] = urljoin(current_url, task["submit_url"])
            logger.info(f"Parsed task: {task}")

            question = task.get("question", "")
            data_url = task.get("data_url")
            submit_url = task.get("submit_url") or urljoin(current_url, "/submit")

            answer = None

            # ---------------------------
            # If data_url is CSV -> deterministic sum logic (no LLM)
            # ---------------------------
            if data_url and data_url.lower().endswith(".csv"):
                fp = download_file(data_url)
                if fp:
                    try:
                        df = pd.read_csv(fp)
                        # Heuristic: look for column named 'value' (case-insensitive)
                        col_candidates = [c for c in df.columns if c.lower() in ("value","amount","sum","total","score","count")]
                        if col_candidates:
                            col = col_candidates[0]
                            # compute sum deterministically
                            numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
                            summ = numeric.sum()
                            # If question asks for integer, make int
                            if float(summ).is_integer():
                                answer = int(summ)
                            else:
                                answer = float(summ)
                        else:
                            # fallback: sum all numeric columns
                            numeric_df = df.select_dtypes(include=["number"])
                            if not numeric_df.empty:
                                sums = numeric_df.sum().sum()
                                answer = int(sums) if float(sums).is_integer() else float(sums)
                            else:
                                # fallback to small-sample LLM if no numeric data
                                preview = df.head(30).to_string()
                                answer = ask_gemini(f"Data:\n{preview}\n\nQuestion: {question}\nReturn only the numerical result.")
                    except Exception as e:
                        logger.warning(f"CSV handling failed: {e}")
                        answer = ask_gemini(f"Unable to parse CSV reliably. Question: {question}", content=html[:20000])

            # ---------------------------
            # PDF -> try tabula -> deterministic
            # ---------------------------
            elif data_url and data_url.lower().endswith(".pdf"):
                fp = download_file(data_url)
                if fp:
                    try:
                        dfs = tabula.read_pdf(fp, pages="all")
                        if dfs and len(dfs) > 0:
                            df0 = dfs[0]
                            numeric_df = df0.select_dtypes(include=["number"])
                            if not numeric_df.empty:
                                s = numeric_df.sum().sum()
                                answer = int(s) if float(s).is_integer() else float(s)
                            else:
                                # convert columns to numeric where possible and sum 'value' column if present
                                cols = df0.columns
                                for c in cols:
                                    try:
                                        df0[c] = pd.to_numeric(df0[c], errors="coerce")
                                    except:
                                        pass
                                if 'value' in map(str.lower, map(str, cols)):
                                    # find that column case-insensitive
                                    idx = [i for i,c in enumerate(cols) if str(c).lower()=='value'][0]
                                    colname = cols[idx]
                                    s = pd.to_numeric(df0[colname], errors="coerce").fillna(0).sum()
                                    answer = int(s) if float(s).is_integer() else float(s)
                                else:
                                    answer = ask_gemini(f"PDF table:\n{df0.head(20).to_string()}\nQuestion: {question}\nReturn only the result.")
                        else:
                            answer = ask_gemini(f"Question: {question}", content=html[:20000])
                    except Exception as e:
                        logger.warning(f"Tabula failed: {e}")
                        answer = ask_gemini(f"Question: {question}", content=html[:20000])

            # ---------------------------
            # Generic file or page -> heuristic extraction
            # ---------------------------
            elif data_url:
                fp = download_file(data_url)
                if fp:
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read(20000)
                    except:
                        txt = ""
                    # Try simple secret extraction
                    candidate = extract_secret_from_text(txt)
                    # If candidate is short and numeric/alnum, use it
                    if candidate and re.fullmatch(r"[A-Za-z0-9_\-]{4,80}", candidate):
                        answer = candidate
                    else:
                        # fallback LLM
                        llm = ask_gemini(f"QUESTION: {question}\n\nFILE CONTENT:\n{txt}\n\nReturn ONLY the required answer.")
                        cleaned = clean_llm_output(llm)
                        # if JSON returned, extract answer field
                        js = try_parse_json_if_present(cleaned)
                        if isinstance(js, dict) and "answer" in js:
                            answer = js["answer"]
                        else:
                            answer = cleaned

            else:
                # No data_url: try to extract from page HTML deterministically
                # Look for 'secret' label first
                secret = extract_secret_from_text(html)
                if secret and len(secret) >= 4:
                    answer = secret
                else:
                    llm = ask_gemini(f"Question: {question}\n\nHTML:\n{html[:15000]}\nReturn only the answer.")
                    cleaned = clean_llm_output(llm)
                    js = try_parse_json_if_present(cleaned)
                    if isinstance(js, dict) and "answer" in js:
                        answer = js["answer"]
                    else:
                        answer = cleaned

            # final cleanup of answer
            if isinstance(answer, str):
                answer = answer.strip()
                # if looks like a JSON payload, try to extract the 'answer' value
                js = try_parse_json_if_present(answer)
                if isinstance(js, dict) and "answer" in js:
                    answer = js["answer"]

            logger.info(f"Computed Answer: {repr(answer)}")

            # Submit
            payload = {
                "email": MY_EMAIL,
                "secret": MY_SECRET,
                "url": current_url,
                "answer": answer
            }
            logger.info(f"Submitting payload to {submit_url}: {json.dumps({'email':payload['email'],'secret':'****','url':payload['url'],'answer':str(payload['answer'])})}")
            try:
                res = requests.post(submit_url, json=payload, timeout=20)
                res.raise_for_status()
                res_json = res.json()
            except Exception as e:
                logger.error(f"Submit failed: {e}")
                break

            logger.info(f"Submission Response: {res_json}")

            if res_json.get("correct"):
                # move to next url if provided
                next_url = res_json.get("url")
                if next_url:
                    current_url = next_url
                    logger.info("Answer accepted. Moving to next URL.")
                    # respect optional delay provided by server
                    delay = res_json.get("delay")
                    if delay and isinstance(delay, (int,float)):
                        logger.info(f"Sleeping for {delay}s as requested.")
                        time.sleep(float(delay))
                    break  # continue main while loop
                else:
                    logger.info("Answer correct and no new URL provided. Quiz finished successfully.")
                    current_url = None
                    break
            else:
                # incorrect - server may return next url or not
                logger.warning(f"Answer incorrect. Reason: {res_json.get('reason')}")
                # server sometimes returns a next URL even when incorrect
                current_url = res_json.get("url")
                if current_url:
                    logger.info("Server provided a next URL despite incorrect answer. Continuing.")
                    # respect delay if provided
                    delay = res_json.get("delay")
                    if delay and isinstance(delay, (int,float)):
                        time.sleep(float(delay))
                    continue
                else:
                    logger.info("No next URL; ending quiz loop.")
                    current_url = None
                    break

        # end for attempts

    # end while
    if not current_url:
        logger.info("END OF QUIZ: No more URLs. Quiz loop completed.")
    elif time.time() >= deadline:
        logger.info("END OF QUIZ: Deadline reached (time limit).")
    logger.info("Background solver exiting.")
    return

# --------------------------------------------------------------------
# API ENDPOINT (validate secret and return 200 immediately)
# --------------------------------------------------------------------

@app.post("/")
async def receive_task(task: QuizRequest, background_tasks: BackgroundTasks):
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    # start background solver
    background_tasks.add_task(solve_quiz_loop, task.url)
    # respond immediately 200 OK after validating secret
    return {"message": "Processing started", "email": task.email, "url": task.url}