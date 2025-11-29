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
    """Scrapes dynamic HTML using Playwright with Smart Waits"""
    logger.info(f"Scraping: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        
        # Try to wait for the question or secret to appear
        try:
            page.wait_for_selector("body", timeout=5000)
            # Short wait for JS execution
            page.wait_for_timeout(2000)
        except:
            pass
            
        content = page.content()
        # Also get visible text to find "Secret code is..." easily
        visible_text = page.inner_text("body")
        browser.close()
        return content, visible_text

def download_file(file_url):
    """Downloads a file to /tmp/"""
    try:
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
    If multiple files exist, get ALL URLs separated by commas in 'data_url'.
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
    # Find all likely data links
    links = []
    for m in re.finditer(r'href=["\'](.*?\.(csv|opus|mp3|pdf))["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    return ",".join(list(set(links)))

# --- CORE SOLVER LOGIC ---

def solve_quiz_loop(start_url):
    current_url = start_url
    
    while current_url:
        try:
            logger.info(f"--- Processing Level: {current_url} ---")
            
            # 1. Scrape Page & Visible Text
            html, visible_text = get_page_content(current_url)
            
            # FAST PATH: Check if secret is already visible (Level 2 Fix)
            secret_match = re.search(r'Secret code is[:\s]*([A-Za-z0-9]+)', visible_text)
            if secret_match:
                logger.info("Found secret in visible text!")
                task = parse_quiz_page(html) # Get submit URL
                answer = secret_match.group(1)
                # Skip download/solve steps
            
            else:
                # 2. Standard Analyze
                task = parse_quiz_page(html)
                
                # Fix URLs
                if task.get("submit_url"): task["submit_url"] = urljoin(current_url, task["submit_url"])
                
                # Manual Link Hunt if Gemini missed them
                manual_links = manual_url_extraction(html, current_url)
                if manual_links and not task.get("data_url"):
                    task["data_url"] = manual_links
                elif manual_links and task.get("data_url"):
                    # Merge them
                    task["data_url"] = task["data_url"] + "," + manual_links

                task["question"] = global_sanitizer(task["question"])
                logger.info(f"Task Parsed: {task}")
                
                # 3. Process Assets
                answer = None
                context_info = ""
                csv_file = None

                # Handle multiple files (Audio + CSV)
                data_urls = [u.strip() for u in str(task.get("data_url", "")).split(",") if u.strip()]
                
                for d_url in data_urls:
                    full_url = urljoin(current_url, d_url)
                    f_path = download_file(full_url)
                    if not f_path: continue

                    if f_path.endswith(('.opus', '.mp3', '.wav')):
                        # Audio: Transcribe/Extract info
                        transcription = ask_gemini_audio(f"Transcribe this audio. Extract any cutoff values or instructions relevant to: {task['question']}", f_path)
                        context_info += f"\nAUDIO TRANSCRIPT: {transcription}\n"
                    
                    elif f_path.endswith('.csv'):
                        csv_file = f_path
                    
                    elif f_path.endswith('.pdf'):
                        try:
                            dfs = tabula.read_pdf(f_path, pages='all')
                            if dfs: context_info += f"\nPDF TABLE: {dfs[0].to_string()}\n"
                        except: pass

                # 4. SOLVE
                if csv_file:
                    # Smart Pandas Solver (Level 3 Fix)
                    try:
                        df = pd.read_csv(csv_file)
                        # Reload with header=None if numeric headers detected
                        if any(str(col).replace('.','',1).isdigit() for col in df.columns):
                            df = pd.read_csv(csv_file, header=None)
                            # Rename columns to A, B, C...
                            df.columns = [chr(65+i) for i in range(len(df.columns))]

                        preview = df.head().to_string()
                        
                        # Ask Gemini for Python Logic
                        logic_prompt = (
                            f"Data Preview:\n{preview}\n\n"
                            f"Question: {task['question']}\n"
                            f"Additional Context: {context_info}\n\n"
                            f"Write a Python expression to calculate the answer using dataframe `df`.\n"
                            f"Examples: `df['A'].sum()`, `df[df['A'] > 50]['B'].sum()`\n"
                            f"Return ONLY the expression string."
                        )
                        expression = ask_gemini(logic_prompt).replace("```python", "").replace("```", "").strip()
                        logger.info(f"Executing Expression: {expression}")
                        
                        # Execute (Safe-ish eval)
                        result = eval(expression, {"df": df, "np": np})
                        answer = int(result) if hasattr(result, 'real') else str(result)
                    except Exception as e:
                        logger.error(f"Pandas Logic Failed: {e}")
                        # Fallback: Total Sum
                        answer = int(df.select_dtypes(include=[np.number]).sum().sum())

                elif context_info:
                    # Just Audio/PDF/Text
                    solve_prompt = f"Question: {task['question']}\nContext: {context_info}\nReturn ONLY the answer."
                    answer = ask_gemini(solve_prompt)
                
                else:
                    # Generic / Text only
                    answer = ask_gemini(f"Question: {task['question']}\nHTML: {visible_text[:1000]}")

            # CLEANUP
            try:
                clean_ans = str(answer).strip().replace("**", "").replace("`", "").replace('"', '').replace("'", "")
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
            res_json = res.json()
            logger.info(f"Result: {res_json}")

            if res_json.get("correct"):
                current_url = res_json.get("url")
            else:
                current_url = res_json.get("url") # Force move to next if available, or loop ends
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