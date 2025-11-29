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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MY_SECRET = os.getenv("MY_SECRET", "default_secret")
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

# --- IMPROVED SOLVER LOGIC WITH RETRY ---

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
            
            # Inner loop for attempts (Max 2 attempts: Initial + 1 Retry)
            for attempt in range(2): 
                answer = None
                
                # --- SOLVING LOGIC ---
                # (Re-use the logic from before, but allowing for a 'retry' prompt)
                
                if task.get("data_url"):
                    file_path = download_file(task["data_url"])
                    
                    if file_path.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        data_preview = df.head().to_string() + "\nColumns: " + str(df.columns.tolist())
                        
                        # If retrying, ask to be more careful
                        tone = "Double check your calculation." if attempt > 0 else "Return ONLY the result."
                        math_prompt = f"Given this CSV data: \n{data_preview}\n\nQuestion: {task['question']}. {tone}"
                        answer = ask_gemini(math_prompt)

                    elif file_path.endswith(".pdf"):
                        dfs = tabula.read_pdf(file_path, pages='all')
                        if dfs:
                            table_str = dfs[0].to_string()
                            tone = "Verify the numbers carefully." if attempt > 0 else "Return ONLY the result."
                            math_prompt = f"PDF Table: \n{table_str}\n\nQuestion: {task['question']}. {tone}"
                            answer = ask_gemini(math_prompt)
                else:
                    tone = "Think step by step." if attempt > 0 else "Answer briefly."
                    answer = ask_gemini(f"Question: {task['question']}\n{tone}")

                # Clean Answer Logic
                try:
                    clean_ans = str(answer).strip()
                    # Remove markdown formatting if present
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
                    # Success! Follow the NEW url immediately
                    current_url = res_json.get("url")
                    break # Break inner attempt loop, go to next level
                
                else:
                    # Wrong Answer
                    logger.warning("Answer incorrect.")
                    
                    if attempt == 0:
                        # If we have a retry left, continue the inner loop to try again
                        logger.info("Retrying current level...")
                        continue 
                    else:
                        # If we are out of retries, we MUST accept the server's next_url (if provided)
                        # The user brief says: "url in your last response is the one you must follow"
                        current_url = res_json.get("url")
                        break # Break inner loop, move on

            # End of While Loop safety check
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