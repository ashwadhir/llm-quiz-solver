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

# Use the STABLE model name
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
        page.wait_for_timeout(4000) 
        content = page.content()
        browser.close()
        return content

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
    """Aggressively removes the User ID and Email"""
    if not isinstance(text_input, str): return text_input
    text = text_input
    if MY_EMAIL:
        text = text.replace(MY_EMAIL, "[REDACTED_EMAIL]")
    text = text.replace(MY_ID_STRING, "[REDACTED_ID]")
    return text

def ask_gemini(prompt, content=""):
    """Sends a request to Gemini"""
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
        if response.parts:
            return response.text.strip()
        else:
            return ""
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
        return "0"

def parse_quiz_page(html_content):
    # Regex fallback for question
    question_text = "Calculate the sum."
    match = re.search(r'(Q\d+\.|Question:)(.{1,300})', html_content, re.IGNORECASE)
    if match:
        question_text = match.group(0)

    prompt = f"""
    Analyze this HTML. Extract the JSON task.
    JSON Format:
    {{
        "question": "The exact question text.",
        "data_url": "URL of file to download (or null)",
        "submit_url": "URL to POST answer to",
    }}
    Return ONLY raw JSON.
    """
    cleaned_text = ask_gemini(prompt, html_content[:50000]) 
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned_text)
        if not data.get("question") or "Extract" in data.get("question"):
             data["question"] = question_text
        return data
    except:
        return {"question": question_text, "data_url": None, "submit_url": None}

def manual_url_extraction(html_content, current_url):
    """Fallback to find links if Gemini fails"""
    # Look for common data patterns
    patterns = [
        r'href=["\'](.*?.csv)["\']',
        r'href=["\'](.*?.opus)["\']',
        r'href=["\'](.*?.mp3)["\']',
        r'href=["\'](.*?.pdf)["\']',
        r'href=["\'](.*?-data.*?)["\']' # Catches /demo-scrape-data
    ]
    for p in patterns:
        match = re.search(p, html_content)
        if match:
            found_url = match.group(1)
            # Ignore purely relative empty links
            if len(found_url) > 2:
                return urljoin(current_url, found_url)
    return None

# --- CORE SOLVER LOGIC ---

def solve_quiz_loop(start_url):
    current_url = start_url
    
    while current_url:
        try:
            logger.info(f"--- Processing Level: {current_url} ---")
            
            # 1. Scrape Page
            html = get_page_content(current_url)
            
            # 2. Analyze with Gemini
            task = parse_quiz_page(html)
            
            # --- URL FIXING LOGIC ---
            if task.get("data_url"):
                task["data_url"] = urljoin(current_url, task["data_url"])
            
            if task.get("submit_url"):
                task["submit_url"] = urljoin(current_url, task["submit_url"])
            
            # FALLBACK: If Gemini missed the data URL, find it manually
            if not task.get("data_url"):
                logger.info("Gemini missed data_url. Attempting manual regex...")
                manual_link = manual_url_extraction(html, current_url)
                if manual_link:
                    logger.info(f"Manual regex found: {manual_link}")
                    task["data_url"] = manual_link

            if not task.get("question"):
                task["question"] = "Calculate the sum of the numbers."

            task["question"] = global_sanitizer(task["question"])
            logger.info(f"Task Parsed: {task}")
            
            for attempt in range(2): 
                answer = None
                
                # --- SOLVING LOGIC ---
                if task.get("data_url"):
                    file_path = download_file(task["data_url"])
                    
                    if not file_path:
                        answer = "Error downloading"
                    
                    # 1. AUDIO FILES
                    elif file_path.endswith(('.opus', '.mp3', '.wav', '.ogg')):
                        math_prompt = f"Listen to this audio. {task['question']}. If asked for a sum, extract the numbers and sum them. Return ONLY the result."
                        answer = ask_gemini_audio(math_prompt, file_path)

                    # 2. CSV FILES
                    elif file_path.endswith(".csv"):
                        try:
                            df = pd.read_csv(file_path)
                            if any(str(col).replace('.','',1).isdigit() for col in df.columns):
                                df = pd.read_csv(file_path, header=None)

                            numeric_df = df.select_dtypes(include=[np.number])
                            total_sum = numeric_df.sum().sum()
                            answer = int(total_sum)
                        except:
                            answer = 0

                    # 3. PDF FILES
                    elif file_path.endswith(".pdf"):
                        try:
                            dfs = tabula.read_pdf(file_path, pages='all')
                            if dfs:
                                table_str = dfs[0].to_string()
                                math_prompt = f"PDF Table:\n{table_str}\n\nQuestion: {task['question']}\nReturn ONLY the result."
                                answer = ask_gemini(math_prompt)
                            else:
                                answer = "0"
                        except:
                            answer = "0"

                    # 4. GENERIC FILES
                    else:
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                raw_content = f.read(8000)
                        except:
                            raw_content = ""

                        raw_content = global_sanitizer(raw_content)

                        # Recursive Script Chaser
                        script_match = re.search(r'<script.*?src=["\'](.*?)["\'].*?>', raw_content)
                        if script_match:
                            script_name = script_match.group(1)
                            script_url = urljoin(task["data_url"], script_name)
                            try:
                                js_content = requests.get(script_url, timeout=5).text
                                raw_content += f"\n\n--- LINKED SCRIPT ({script_name}) ---\n{js_content}"
                            except: pass

                        import_match = re.search(r'import.*?from\s+["\'](.*?)["\']', raw_content)
                        if import_match:
                            import_name = import_match.group(1)
                            import_url = urljoin(task["data_url"], import_name)
                            try:
                                imported_content = requests.get(import_url, timeout=5).text
                                imported_content = global_sanitizer(imported_content)
                                raw_content += f"\n\n--- IMPORTED MODULE ({import_name}) ---\n{imported_content}"
                            except: pass

                        extraction_prompt = (
                            f"QUESTION: {task['question']}\n"
                            f"CONTENT: {raw_content}\n"
                            f"TASK: Extract the secret code. It is NOT '[REDACTED_ID]'."
                        )
                        answer = ask_gemini(extraction_prompt)
                        
                        if not answer or "[" in str(answer):
                            cleanr = re.compile('<.*?>')
                            answer = re.sub(cleanr, ' ', raw_content).strip()

                else:
                    tone = "Think step by step." if attempt > 0 else "Answer directly."
                    answer = ask_gemini(f"Question: {task['question']}\n{tone}", content=html[:20000])

                try:
                    clean_ans = str(answer).strip()
                    clean_ans = clean_ans.replace("**", "").replace("`", "").replace('"', '').replace("'", "")
                    if "secret is" in clean_ans.lower():
                        clean_ans = clean_ans.split("secret is")[-1].strip()
                    
                    if MY_ID_STRING in clean_ans: clean_ans = ""
                    
                    if clean_ans.replace('.','',1).isdigit():
                        answer = float(clean_ans) if '.' in clean_ans else int(clean_ans)
                    else:
                        answer = clean_ans
                except:
                    pass

                payload = {
                    "email": MY_EMAIL,
                    "secret": MY_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                if not task.get("submit_url"):
                    task["submit_url"] = urljoin(current_url, "/submit")

                logger.info(f"Submitting: {payload}")
                res = requests.post(task["submit_url"], json=payload)
                res_json = res.json()
                logger.info(f"Result: {res_json}")

                if res_json.get("correct"):
                    current_url = res_json.get("url")
                    break 
                else:
                    if attempt == 0: continue 
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