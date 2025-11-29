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
# Hardcoded ID to scrub (extracted from your logs)
MY_ID_STRING = "22f2000771" 

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
        if not filename: filename = "downloaded_data"
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
    """Aggressively removes the User ID and Email from ANY string input"""
    if not isinstance(text_input, str): return text_input
    text = text_input
    # Remove Email
    if MY_EMAIL:
        text = text.replace(MY_EMAIL, "[REDACTED_EMAIL]")
    # Remove ID specifically
    text = text.replace(MY_ID_STRING, "[REDACTED_ID]")
    return text

def ask_gemini(prompt, content=""):
    """Sends a request to Gemini 2.5 Flash"""
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
    
    # Sanitize Prompt and Context before sending
    prompt = global_sanitizer(prompt)
    content = global_sanitizer(content)

    system_instruction = (
        "You are a precise data extraction engine. "
        "Output ONLY the requested value. "
        "Do NOT return the user's email or ID."
    )
    full_prompt = f"{system_instruction}\n\nCONTEXT:\n{content}\n\nTASK:\n{prompt}"
    
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
    """Uploads audio to Gemini and asks question"""
    logger.info(f"Processing Audio: {audio_path}")
    try:
        # Upload the file
        audio_file = genai.upload_file(path=audio_path)
        
        # Wait for processing (usually instant for small files)
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, audio_file])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Audio Error: {e}")
        return "0"

def parse_quiz_page(html_content):
    """Uses Gemini to extract JSON instructions from HTML"""
    
    # regex fallback
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

def clean_html_text(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return ' '.join(cleantext.split())

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
            
            if task.get("data_url"):
                task["data_url"] = urljoin(current_url, task["data_url"])
            if task.get("submit_url"):
                task["submit_url"] = urljoin(current_url, task["submit_url"])
            
            if not task.get("question"):
                task["question"] = "Calculate the sum of the numbers."

            # SANITIZE TASK DICTIONARY
            task["question"] = global_sanitizer(task["question"])
            
            logger.info(f"Task Parsed: {task}")
            
            for attempt in range(2): 
                answer = None
                
                # --- SOLVING LOGIC ---
                if task.get("data_url"):
                    file_path = download_file(task["data_url"])
                    
                    if not file_path:
                        answer = "Error downloading"
                    
                    # 1. AUDIO FILES (.opus, .mp3, .wav)
                    elif file_path.endswith(('.opus', '.mp3', '.wav', '.ogg')):
                        math_prompt = f"Listen to this audio. {task['question']}. If asked for a sum, extract the numbers and sum them. Return ONLY the result."
                        answer = ask_gemini_audio(math_prompt, file_path)

                    # 2. CSV FILES (Python Math)
                    elif file_path.endswith(".csv"):
                        try:
                            df = pd.read_csv(file_path)
                            numeric_df = df.select_dtypes(include=[np.number])
                            total_sum = numeric_df.sum().sum()
                            logger.info(f"Python Calculated Sum: {total_sum}")
                            answer = int(total_sum) 
                        except:
                            answer = 0

                    # 3. PDF FILES (Tabula)
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

                    # 4. GENERIC FILES (Script/HTML)
                    else:
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                raw_content = f.read(8000)
                        except:
                            raw_content = ""

                        # Check for linked scripts (Improved Regex)
                        script_match = re.search(r'<script.*?src=["\'](.*?)["\'].*?>', raw_content)
                        if script_match:
                            script_name = script_match.group(1)
                            script_url = urljoin(task["data_url"], script_name)
                            try:
                                js_content = requests.get(script_url, timeout=5).text
                                raw_content += f"\n\n--- SCRIPT CONTENT ---\n{js_content}"
                            except Exception as e:
                                logger.error(f"Failed to download script: {e}")

                        extraction_prompt = (
                            f"QUESTION: {task['question']}\n"
                            f"CONTENT: {raw_content}\n"
                            f"TASK: Extract the secret code. It is NOT '[REDACTED_ID]'."
                        )
                        answer = ask_gemini(extraction_prompt)
                        
                        if not answer:
                            answer = clean_html_text(raw_content).strip()
                        
                        # DOUBLE CHECK: If answer is your ID, block it
                        if MY_ID_STRING in str(answer):
                            logger.warning("Agent attempted to submit ID. Blocking.")
                            answer = ""

                else:
                    # Pure Text Question
                    tone = "Think step by step." if attempt > 0 else "Answer directly."
                    answer = ask_gemini(f"Question: {task['question']}\n{tone}", content=html[:20000])

                # Clean Answer Logic
                try:
                    clean_ans = str(answer).strip()
                    clean_ans = clean_ans.replace("**", "").replace("`", "").replace('"', '').replace("'", "")
                    if "secret is" in clean_ans.lower():
                        clean_ans = clean_ans.split("secret is")[-1].strip()
                    
                    if clean_ans.replace('.','',1).isdigit():
                        answer = float(clean_ans) if '.' in clean_ans else int(clean_ans)
                    else:
                        answer = clean_ans
                except:
                    pass

                # 3. Submit
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
                
                try:
                    res_json = res.json()
                except:
                    res_json = {}

                logger.info(f"Result: {res_json}")

                if res_json.get("correct"):
                    current_url = res_json.get("url")
                    break 
                else:
                    logger.warning("Answer incorrect.")
                    if attempt == 0:
                        logger.info("Retrying...")
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