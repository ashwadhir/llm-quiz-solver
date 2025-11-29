import os
import json
import requests
import logging
import pandas as pd
import tabula
import google.generativeai as genai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin  # <--- NEW IMPORT

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
        page.wait_for_selector("body") 
        content = page.content()
        browser.close()
        return content

def download_file(file_url):
    """Downloads a file to /tmp/ and returns local path"""
    # Sanitize filename to remove query parameters for saving
    filename = file_url.split("/")[-1].split("?")[0]
    local_filename = f"/tmp/{filename}"
    
    logger.info(f"Downloading {file_url} to {local_filename}")
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def ask_gemini(prompt, content=""):
    """Sends a request to gemini-2.5-flash"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    full_prompt = f"{prompt}\n\nContext:\n{content}"
    response = model.generate_content(full_prompt)
    return response.text

def parse_quiz_page(html_content):
    """Uses Gemini to extract JSON instructions from HTML"""
    prompt = """
    Analyze this HTML. Extract the following JSON:
    {
        "question": "The exact question text",
        "data_url": "The URL of the file to download (or null)",
        "submit_url": "The URL to POST the answer to",
        "format": "The expected answer format (number, string, etc.)"
    }
    Return ONLY raw JSON.
    """
    cleaned_text = ask_gemini(prompt, html_content[:25000]) 
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_text)

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
            
            # --- CRITICAL FIX: Handle Relative URLs ---
            if task.get("data_url"):
                task["data_url"] = urljoin(current_url, task["data_url"])
            
            if task.get("submit_url"):
                task["submit_url"] = urljoin(current_url, task["submit_url"])
                
            logger.info(f"Task Parsed: {task}")
            
            # Inner loop for attempts (Max 2 attempts)
            for attempt in range(2): 
                answer = None
                
                # --- SOLVING LOGIC ---
                if task.get("data_url"):
                    file_path = download_file(task["data_url"])
                    
                    if file_path.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        data_preview = df.head().to_string() + "\nColumns: " + str(df.columns.tolist())
                        
                        tone = "Double check your calculation." if attempt > 0 else "Return ONLY the result."
                        math_prompt = f"Given this CSV data: \n{data_preview}\n\nQuestion: {task['question']}. {tone}"
                        answer = ask_gemini(math_prompt)

                    elif file_path.endswith(".pdf"):
                        # Use Tabula to extract tables
                        dfs = tabula.read_pdf(file_path, pages='all')
                        if dfs:
                            table_str = dfs[0].to_string()
                            tone = "Verify the numbers carefully." if attempt > 0 else "Return ONLY the result."
                            math_prompt = f"PDF Table: \n{table_str}\n\nQuestion: {task['question']}. {tone}"
                            answer = ask_gemini(math_prompt)
                    else:
                        # Fallback for text files or scraping
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            file_content = f.read(2000) # Read first 2k chars
                        answer = ask_gemini(f"File content: {file_content}\n\nQuestion: {task['question']}")

                else:
                    tone = "Think step by step." if attempt > 0 else "Answer briefly."
                    answer = ask_gemini(f"Question: {task['question']}\n{tone}")

                # Clean Answer Logic
                try:
                    clean_ans = str(answer).strip()
                    clean_ans = clean_ans.replace("**", "").replace("`", "")
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
                logger.info(f"Submission Attempt {attempt+1}: {payload}")
                
                res = requests.post(task["submit_url"], json=payload)
                res_json = res.json()
                logger.info(f"Result: {res_json}")

                # 4. DECISION MATRIX
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