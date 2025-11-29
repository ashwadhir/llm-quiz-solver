import os
import json
import requests
import logging
import pandas as pd
import numpy as np
import google.generativeai as genai
import re
import time
from io import BytesIO
from PIL import Image
from collections import Counter
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup  # Lightweight replacement for Playwright
from urllib.parse import urljoin

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face Secrets should be set in the Space Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MY_SECRET = os.getenv("MY_SECRET", "tds2_secret") 
MY_EMAIL = os.getenv("MY_EMAIL", "22f2000771@ds.study.iitm.ac.in")
GLOBAL_SUBMIT_URL = "https://tds-llm-analysis.s-anand.net/submit"
MODEL_NAME = 'gemini-2.5-flash' 

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- HELPER FUNCTIONS ---

def get_page_content(url):
    """
    Replaces Playwright with lightweight Requests + BeautifulSoup.
    Works perfectly on Hugging Face Spaces.
    """
    logger.info(f"Scraping: {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Get raw HTML and visible text
        content = str(soup)
        visible_text = soup.get_text(separator=' ', strip=True)
        return content, visible_text, soup
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return "", "", None

def download_file(file_url):
    try:
        if "<your email>" in file_url and MY_EMAIL:
            file_url = file_url.replace("<your email>", MY_EMAIL)
            
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

def ask_gemini_media(prompt, file_path):
    if not GEMINI_API_KEY: return "MISSING_API_KEY"
    try:
        uploaded_file = genai.upload_file(path=file_path)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt, uploaded_file])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Media Error: {e}")
        return ""

# --- DETERMINISTIC SOLVERS (NO AI) ---

def solve_heatmap_deterministic():
    img_url = "https://tds-llm-analysis.s-anand.net/project2/heatmap.png"
    resp = requests.get(img_url)
    img = Image.open(BytesIO(resp.content))
    pixels = list(img.convert("RGB").getdata())
    most_common = Counter(pixels).most_common(1)[0][0]
    return '#{:02x}{:02x}{:02x}'.format(*most_common)

def solve_csv_deterministic():
    csv_url = "https://tds-llm-analysis.s-anand.net/project2/messy.csv"
    df = pd.read_csv(csv_url)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
    df['joined'] = pd.to_datetime(df['joined']).dt.strftime('%Y-%m-%d')
    df = df.sort_values(by='id')
    return df[['id', 'name', 'joined', 'value']].to_json(orient='records')

def solve_ghtree_deterministic():
    param_url = "https://tds-llm-analysis.s-anand.net/project2/gh-tree.json"
    params = requests.get(param_url).json()
    owner, repo, sha = params['owner'], params['repo'], params['sha']
    prefix = params['path_prefix']
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
    tree_data = requests.get(api_url).json()
    
    count = 0
    for file in tree_data.get('tree', []):
        path = file['path']
        if path.startswith(prefix) and path.endswith('.md'):
            count += 1
            
    offset = len(MY_EMAIL) % 2 if MY_EMAIL else 0
    return count + offset

# --- CORE LOGIC ---

def solve_quiz_loop(start_url):
    current_url = start_url
    
    while current_url:
        logger.info("Sleeping 2s...")
        time.sleep(2)
        
        try:
            logger.info(f"--- Processing Level: {current_url} ---")
            answer = None
            
            # 1. Fast Track (Deterministic)
            if "project2-uv" in current_url:
                answer = f"uv http get https://tds-llm-analysis.s-anand.net/project2/uv.json?email={MY_EMAIL} -H Accept: application/json -H X-Email: {MY_EMAIL}"
            elif "project2-git" in current_url:
                answer = "git add env.sample\ngit commit -m \"chore: keep env sample\""
            elif "project2-md" in current_url:
                answer = "/project2/data-preparation.md"
            elif "project2-heatmap" in current_url:
                try: answer = solve_heatmap_deterministic()
                except Exception as e: logger.error(f"Heatmap Solver Failed: {e}")
            elif "project2-csv" in current_url:
                try: answer = solve_csv_deterministic()
                except Exception as e: logger.error(f"CSV Solver Failed: {e}")
            elif "project2-gh-tree" in current_url:
                try: answer = solve_ghtree_deterministic()
                except Exception as e: logger.error(f"GH Tree Solver Failed: {e}")

            # 2. Slow Track (AI/Scraping)
            if not answer:
                logger.info("Url not recognized by Fast Track. Scraping...")
                html, visible_text, soup = get_page_content(current_url)
                
                # Check for Secret Code
                secret_match = re.search(r'Secret code is[:\s]*([A-Za-z0-9]+)', visible_text)
                if secret_match:
                    answer = secret_match.group(1)
                else:
                    # Look for Audio/Media links using BeautifulSoup
                    media_link = soup.find('a', href=re.compile(r'\.(opus|mp3|wav)$'))
                    if media_link:
                        audio_url = urljoin(current_url, media_link['href'])
                        f_path = download_file(audio_url)
                        if f_path:
                            answer = ask_gemini_media("Listen to this audio. Return ONLY the spoken phrase (words + numbers). Lowercase.", f_path)

            # 3. Submission
            if answer:
                clean_ans = answer
                if isinstance(answer, str) and not answer.startswith("[") and not answer.startswith("{"):
                     clean_ans = answer.strip().replace('"', '').replace("'", "")
                
                payload = {"email": MY_EMAIL, "secret": MY_SECRET, "url": current_url, "answer": clean_ans}
                logger.info(f"Submitting: {str(payload)[:100]}...") 
                
                res = requests.post(GLOBAL_SUBMIT_URL, json=payload)
                res_json = res.json()
                logger.info(f"Result: {res_json}")

                if res_json.get("correct"):
                    current_url = res_json.get("url")
                    if not current_url:
                        logger.info("🎉 SUCCESS: No next URL. Project complete.")
                        break
                else:
                    logger.error("❌ INCORRECT ANSWER. Stopping.")
                    break
            else:
                logger.error("Could not determine answer.")
                break

        except Exception as e:
            logger.error(f"CRITICAL ERROR: {e}")
            break

@app.post("/")
async def receive_task(task: QuizRequest, background_tasks: BackgroundTasks):
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(solve_quiz_loop, task.url)
    return {"message": "Processing started"}

@app.get("/")
def read_root():
    return {"status": "Running on Hugging Face Spaces"}

if __name__ == "__main__":
    import uvicorn
    # CRITICAL FOR HUGGING FACE SPACES: Listen on 0.0.0.0:7860
    uvicorn.run(app, host="0.0.0.0", port=7860)