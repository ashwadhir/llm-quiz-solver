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
        # Wait for potential hydration
        page.wait_for_timeout(1000) 
        content = page.content()
        browser.close()
        return content

def download_file(file_url):
    """Downloads a file to /tmp/ and returns local path"""
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
    """Sends a request to Gemini 2.5 Flash"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    # System instruction to prevent "Tutorial Mode"
    system_instruction = "You are a precise data extraction engine. You do NOT write code. You do NOT explain. You only output the requested value."
    full_prompt = f"{system_instruction}\n\nCONTEXT:\n{content}\n\nTASK:\n{prompt}"
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return ""

def parse_quiz_page(html_content):
    """Uses Gemini to extract JSON instructions from HTML"""
    prompt = """
    Analyze this HTML. Extract the JSON task.
    
    If you see a question text (like "What is the sum...", "Find the code...", "Download file..."), extract it.
    If the question is implied by a table or list, summarize the task.
    
    JSON Format:
    {
        "question": "The exact question text. If missing, summarize what needs to be done.",
        "data_url": "URL of file to download (or null)",
        "submit_url": "URL to POST answer to",
        "format": "Expected answer type (number, string, json)"
    }
    Return ONLY raw JSON.
    """
    cleaned_text = ask_gemini(prompt, html_content[:40000]) 
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_text)
    except:
        return {"question": "Extract data", "data_url": None, "submit_url": None}

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
            
            # Handle Relative URLs
            if task.get("data_url"):
                task["data_url"] = urljoin(current_url, task["data_url"])
            if task.get("submit_url"):
                task["submit_url"] = urljoin(current_url, task["submit_url"])
            
            # Fail-safe if question is None
            if not task.get("question"):
                task["question"] = "Analyze the data file and extract the key information or result requested."

            logger.info(f"Task Parsed: {task}")
            
            # Inner loop for attempts (Max 2 attempts)
            for attempt in range(2): 
                answer = None
                
                # --- SOLVING LOGIC ---
                if task.get("data_url"):
                    file_path = download_file(task["data_url"])
                    
                    if file_path and file_path.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        # Send first 20 rows + column stats to ensure context
                        data_preview = df.head(20).to_string() + "\n\nColumn Info:\n" + str(df.dtypes)
                        
                        tone = "Be extremely precise with calculations." if attempt > 0 else "Return ONLY the numerical result."
                        math_prompt = f"Data:\n{data_preview}\n\nQuestion: {task['question']}\n\n{tone}"
                        answer = ask_gemini(math_prompt)

                    elif file_path and file_path.endswith(".pdf"):
                        try:
                            dfs = tabula.read_pdf(file_path, pages='all')
                            if dfs:
                                table_str = dfs[0].to_string()
                                tone = "Check the rows carefully." if attempt > 0 else "Return ONLY the result."
                                math_prompt = f"PDF Table:\n{table_str}\n\nQuestion: {task['question']}. {tone}"
                                answer = ask_gemini(math_prompt)
                            else:
                                answer = "0" # Fallback
                        except:
                            answer = "0"

                    elif file_path:
                        # GENERIC FILE (HTML/Text/Scrape Target)
                        # Fix for the "Tutorial Bug"
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            file_content = f.read(5000) 
                        
                        extraction_prompt = (
                            f"I have downloaded the file mentioned in the question.\n"
                            f"The file content is provided above.\n"
                            f"QUESTION: {task['question']}\n"
                            f"TASK: Look at the file content. Extract the EXACT answer string requested (e.g. a secret code, a flag, a name). "
                            f"Do NOT write a script. Do NOT explain how to find it. Just output the value found in the text."
                        )
                        answer = ask_gemini(extraction_prompt, content=file_content)

                else:
                    # Pure Text Question
                    tone = "Think step by step." if attempt > 0 else "Answer directly."
                    answer = ask_gemini(f"Question: {task['question']}\n{tone}")

                # Clean Answer Logic
                try:
                    clean_ans = str(answer).strip()
                    clean_ans = clean_ans.replace("**", "").replace("`", "").replace('"', '').replace("'", "")
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
                
                # Check for valid Submit URL
                if not task.get("submit_url"):
                    logger.error("No submit URL found. Trying default /submit")
                    task["submit_url"] = urljoin(current_url, "/submit")

                res = requests.post(task["submit_url"], json=payload)
                
                try:
                    res_json = res.json()
                except:
                    logger.error(f"Non-JSON response: {res.text}")
                    res_json = {}

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
    # Verify Secret
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    # Start the loop in background
    background_tasks.add_task(solve_quiz_loop, task.url)
    return {"message": "Task processing started"}