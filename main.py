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

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("AIzaSyDCQnwC1bp3A7K3gpUebHaFB4eN0YjrjZs")
MY_SECRET = os.getenv("MY_SECRET", "default_secret")
MY_EMAIL = os.getenv("MY_EMAIL", "your_email@example.com")

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
    local_filename = "/tmp/" + file_url.split("/")[-1]
    logger.info(f"Downloading {file_url} to {local_filename}")
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def ask_gemini(prompt, content=""):
    """Sends a request to Gemini 1.5 Flash"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    full_prompt = f"{prompt}\n\nContext:\n{content}"
    response = model.generate_content(full_prompt)
    return response.text

def parse_quiz_page(html_content):
    """Uses Gemini to extract JSON instructions from HTML"""
    prompt = """
    Analyze this HTML. Extract the following JSON:
    {
        "question": "The exact question text",
        "data_url": "The full URL of the PDF/CSV to download (or null)",
        "submit_url": "The URL to POST the answer to",
        "format": "The expected answer format (number, string, etc.)"
    }
    Return ONLY raw JSON.
    """
    cleaned_text = ask_gemini(prompt, html_content[:20000]) # Send first 20k chars
    # Clean up markdown code blocks if Gemini adds them
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
            logger.info(f"Task: {task}")
            
            answer = None
            
            # 3. Handle Data Files (CSV/PDF)
            if task.get("data_url"):
                file_path = download_file(task["data_url"])
                
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    # Pass the dataframe head/columns to Gemini to figure out the math
                    data_preview = df.head().to_string() + "\nColumns: " + str(df.columns.tolist())
                    
                    math_prompt = f"Given this CSV data preview: \n{data_preview}\n\nAnswer this question: {task['question']}. Return ONLY the result."
                    answer = ask_gemini(math_prompt)

                elif file_path.endswith(".pdf"):
                    # Use Tabula to extract tables
                    dfs = tabula.read_pdf(file_path, pages='all')
                    if dfs:
                        # Convert first table to string for Gemini
                        table_str = dfs[0].to_string()
                        math_prompt = f"Given this PDF table: \n{table_str}\n\nAnswer this question: {task['question']}. Return ONLY the result."
                        answer = ask_gemini(math_prompt)
                    else:
                        logger.error("No tables found in PDF")
                        
            else:
                # 4. Pure Text Question
                answer = ask_gemini(f"Answer this question briefly and directly: {task['question']}")

            # 5. Clean Answer (Remove extra text if strict number required)
            # You might need to add logic here to cast to int/float if the quiz expects numbers
            try:
                # Attempt to convert to simple number if it looks like one
                clean_ans = answer.strip()
                if clean_ans.replace('.','',1).isdigit():
                    answer = float(clean_ans) if '.' in clean_ans else int(clean_ans)
                else:
                    answer = clean_ans
            except:
                pass

            # 6. Submit
            payload = {
                "email": MY_EMAIL,
                "secret": MY_SECRET,
                "url": current_url,
                "answer": answer
            }
            logger.info(f"Submitting: {payload}")
            
            res = requests.post(task["submit_url"], json=payload)
            res_json = res.json()
            logger.info(f"Result: {res_json}")

            # 7. Next Level?
            if res_json.get("correct") and res_json.get("url"):
                current_url = res_json["url"]
            else:
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