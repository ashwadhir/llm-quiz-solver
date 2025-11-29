import os
import json
import requests
import logging
import pandas as pd
import tabula
import google.generativeai as genai
import re  # <--- NEW IMPORT FOR CLEANING HTML
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
        page.wait_for_timeout(2000) 
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

def ask_gemini(prompt, content=""):
    """Sends a request to Gemini 2.5 Flash with Safety Filters DISABLED"""
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
    
    system_instruction = (
        "You are a precise data extraction engine. "
        "Output ONLY the requested value. "
        "Do NOT return the user's email."
    )
    full_prompt = f"{system_instruction}\n\nCONTEXT:\n{content}\n\nTASK:\n{prompt}"
    
    try:
        response = model.generate_content(full_prompt)
        if response.parts:
            return response.text.strip()
        else:
            logger.warning("Gemini returned no text parts.")
            return ""
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return ""

def parse_quiz_page(html_content):
    """Uses Gemini to extract JSON instructions from HTML"""
    prompt = """
    Analyze this HTML. Extract the JSON task.
    JSON Format:
    {
        "question": "The exact question text.",
        "data_url": "URL of file to download (or null)",
        "submit_url": "URL to POST answer to",
    }
    Return ONLY raw JSON.
    """
    cleaned_text = ask_gemini(prompt, html_content[:40000]) 
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_text)
    except:
        return {"question": "Extract data", "data_url": None, "submit_url": None}

def clean_html_text(raw_html):
    """Removes HTML tags to return clean text"""
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
                task["question"] = "Extract the main answer or secret code from the page context."

            logger.info(f"Task Parsed: {task}")
            
            for attempt in range(2): 
                answer = None
                
                # --- SOLVING LOGIC ---
                if task.get("data_url"):
                    file_path = download_file(task["data_url"])
                    
                    if file_path and file_path.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        data_preview = df.head(20).to_string() + "\n\nColumn Info:\n" + str(df.dtypes)
                        math_prompt = f"Data:\n{data_preview}\n\nQuestion: {task['question']}\nReturn ONLY the numerical result."
                        answer = ask_gemini(math_prompt)

                    elif file_path and file_path.endswith(".pdf"):
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

                    elif file_path:
                        # GENERIC FILE (HTML/Text) - ROBUST LOGIC
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                raw_content = f.read(8000)
                        except:
                            raw_content = ""

                        logger.info(f"Downloaded Content Preview: {raw_content[:200]}")
                        
                        # 1. Clean HTML tags to help Gemini
                        text_content = clean_html_text(raw_content)
                        
                        # 2. Ask Gemini with CLEAN text
                        extraction_prompt = (
                            f"QUESTION: {task['question']}\n"
                            f"CONTENT: {text_content}\n"
                            f"TASK: Extract the secret code or answer. Return ONLY the code."
                        )
                        answer = ask_gemini(extraction_prompt)
                        
                        # 3. FALLBACK: If Gemini fails, use the raw text
                        if not answer:
                            logger.warning("Gemini failed. Using raw text fallback.")
                            # Usually the secret is the only significant text in the body
                            answer = text_content.strip()

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