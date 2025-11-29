import os
import json
import requests
import logging
import pandas as pd
import numpy as np
import tabula
import google.generativeai as genai
import re
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MY_SECRET = os.getenv("MY_SECRET", "tds2_secret")
MY_EMAIL = os.getenv("MY_EMAIL", "22f2000771@ds.study.iitm.ac.in")
MY_ID_STRING = "22f2000771" 
MODEL_NAME = 'gemini-2.5-flash'

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- HELPER FUNCTIONS ---

def get_page_content(url):
    """Scrapes dynamic HTML using Playwright"""
    logger.info(f"Scraping: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        try:
            page.wait_for_selector("body", timeout=5000)
            page.wait_for_timeout(2000)
        except: pass
        
        content = page.content()
        visible_text = page.inner_text("body")
        browser.close()
        return content, visible_text

def download_file(file_url):
    """Downloads a file to /tmp/"""
    try:
        # Fix placeholders in URL (e.g. <your email>)
        if "<your email>" in file_url and MY_EMAIL:
            file_url = file_url.replace("<your email>", MY_EMAIL)
        if "YOUR_EMAIL" in file_url and MY_EMAIL:
            file_url = file_url.replace("YOUR_EMAIL", MY_EMAIL)

        filename = file_url.split("/")[-1].split("?")[0]
        if not filename: filename = f"data_{int(time.time())}"
        local_filename = f"/tmp/{filename}"
        
        logger.info(f"Downloading {file_url} to {local_filename}")
        with requests.get(file_url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def global_sanitizer(text_input):
    if not isinstance(text_input, str): return text_input
    text = text_input
    if MY_EMAIL: text = text.replace(MY_EMAIL, "[REDACTED_EMAIL]")
    text = text.replace(MY_ID_STRING, "[REDACTED_ID]")
    return text

def ask_gemini(prompt, content=""):
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    model = genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)
    prompt = global_sanitizer(prompt)
    content = global_sanitizer(content)
    full_prompt = f"CONTEXT:\n{content}\n\nTASK:\n{prompt}"
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip() if response.parts else ""
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return ""

def ask_gemini_audio(prompt, audio_path):
    logger.info(f"Processing Audio: {audio_path}")
    try:
        audio_file = genai.upload_file(path=audio_path)
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt, audio_file])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Audio Error: {e}")
        return ""

def parse_quiz_page(html_content):
    question_text = "Calculate the answer."
    match = re.search(r'(Q\d+\.|Question:)(.{1,300})', html_content, re.IGNORECASE)
    if match: question_text = match.group(0)

    prompt = f"""
    Analyze HTML. Extract task.
    JSON Format: {{ "question": "...", "data_url": "url1,url2", "submit_url": "..." }}
    """
    cleaned_text = ask_gemini(prompt, html_content[:50000]) 
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned_text)
        if not data.get("question"): data["question"] = question_text
        return data
    except:
        return {"question": question_text, "data_url": None, "submit_url": None}

def manual_url_extraction(html_content, current_url):
    links = []
    for m in re.finditer(r'href=["\'](.*?\.(csv|opus|mp3|pdf|json))["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    for m in re.finditer(r'href=["\'](.*?-data.*?)["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    return ",".join(list(set(links)))

# --- CORE SOLVER LOGIC ---

def solve_quiz_loop(start_url):
    current_url = start_url
    
    while current_url:
        try:
            logger.info(f"--- Processing Level: {current_url} ---")
            html, visible_text = get_page_content(current_url)
            
            # FAST PATH: Check visible text for secret
            secret_match = re.search(r'Secret code is[:\s]*([A-Za-z0-9]+)', visible_text)
            if secret_match:
                task = parse_quiz_page(html)
                answer = secret_match.group(1)
            
            else:
                task = parse_quiz_page(html)
                
                # --- URL FIXING LOGIC ---
                if task.get("submit_url"): 
                    task["submit_url"] = urljoin(current_url, task["submit_url"])
                
                # CRITICAL FIX: If submit_url is same as current page, force global submit
                if task.get("submit_url") == current_url:
                    task["submit_url"] = urljoin(current_url, "/submit")
                    logger.info("Fixed Submit URL to global endpoint")
                
                manual_links = manual_url_extraction(html, current_url)
                if manual_links:
                    existing = str(task.get("data_url", ""))
                    task["data_url"] = existing + "," + manual_links if existing else manual_links

                task["question"] = global_sanitizer(task["question"])
                logger.info(f"Task Parsed: {task}")
                
                answer = None
                context_info = ""
                csv_file = None

                data_urls = [u.strip() for u in str(task.get("data_url", "")).split(",") if u.strip()]
                
                for d_url in data_urls:
                    full_url = urljoin(current_url, d_url)
                    f_path = download_file(full_url)
                    if not f_path: continue

                    if f_path.endswith(('.opus', '.mp3', '.wav')):
                        transcription = ask_gemini_audio(f"Transcribe audio. {task['question']}", f_path)
                        context_info += f"\nAUDIO TRANSCRIPT: {transcription}\n"
                    
                    elif f_path.endswith('.csv'):
                        csv_file = f_path
                    
                    elif f_path.endswith('.pdf'):
                        try:
                            dfs = tabula.read_pdf(f_path, pages='all')
                            if dfs: context_info += f"\nPDF TABLE: {dfs[0].to_string()}\n"
                        except: pass
                    
                    else:
                        try:
                            with open(f_path, "r", errors="ignore") as f: content = f.read(8000)
                            content = global_sanitizer(content)
                            context_info += f"\nFILE CONTENT: {content}\n"
                        except: pass

                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        if any(str(col).replace('.','',1).isdigit() for col in df.columns):
                            df = pd.read_csv(csv_file, header=None)
                            df.columns = [chr(65+i) for i in range(len(df.columns))]
                        preview = df.head().to_string()
                        logic_prompt = (
                            f"Data: {preview}\nContext: {context_info}\nQuestion: {task['question']}\n"
                            f"Write Python expression for dataframe `df`.\n"
                            f"If question implies Cutoff, use SUM of values > Cutoff.\n"
                            f"Return ONLY expression."
                        )
                        expression = ask_gemini(logic_prompt).replace("```python", "").replace("```", "").strip()
                        result = eval(expression, {"df": df, "np": np})
                        answer = int(result) if hasattr(result, 'real') else str(result)
                    except:
                        answer = int(df.select_dtypes(include=[np.number]).sum().sum())

                elif context_info:
                    solve_prompt = (
                        f"Question: {task['question']}\nContext: {context_info}\n"
                        f"Extract the answer. Return ONLY the code/command string."
                    )
                    answer = ask_gemini(solve_prompt)
                
                else:
                    answer = ask_gemini(f"Question: {task['question']}\nHTML: {visible_text[:1000]}")

            # CLEANUP (Relaxed for Commands)
            try:
                clean_ans = str(answer).strip().replace("**", "").replace("`", "").replace('"', '').replace("'", "")
                
                # CRITICAL FIX: Only aggressive filter if it looks like a paragraph
                if len(clean_ans) > 150: 
                    code_match = re.search(r'\b([a-zA-Z0-9]{5,20})\b', clean_ans)
                    if code_match: clean_ans = code_match.group(1)

                if "secret is" in clean_ans.lower(): clean_ans = clean_ans.split("secret is")[-1].strip()
                if MY_ID_STRING in clean_ans: clean_ans = ""
                
                if clean_ans.replace('.','',1).isdigit():
                    answer = float(clean_ans) if '.' in clean_ans else int(clean_ans)
                else:
                    answer = clean_ans
            except: pass

            # SUBMIT
            payload = {"email": MY_EMAIL, "secret": MY_SECRET, "url": current_url, "answer": answer}
            if not task.get("submit_url"): task["submit_url"] = urljoin(current_url, "/submit")
            
            logger.info(f"Submitting: {payload}")
            res = requests.post(task["submit_url"], json=payload)
            
            # Handle non-JSON response (404/500 pages)
            try:
                res_json = res.json()
            except:
                logger.error(f"Server returned non-JSON: {res.text[:200]}")
                # Retry strategy could go here, but usually indicates bad URL
                res_json = {}

            logger.info(f"Result: {res_json}")

            if res_json.get("correct"):
                current_url = res_json.get("url")
            else:
                current_url = res_json.get("url")
                if not current_url: break

        except Exception as e:
            logger.error(f"CRITICAL ERROR: {e}")
            break

@app.post("/")
async def receive_task(task: QuizRequest, background_tasks: BackgroundTasks):
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(solve_quiz_loop, task.url)
    return {"message": "Processing"}