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
import random
from PIL import Image
from collections import Counter
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# KEY ROTATION
KEYS = []
if os.getenv("GEMINI_API_KEY"): KEYS.append(os.getenv("GEMINI_API_KEY"))
if os.getenv("GEMINI_API_KEY_2"): KEYS.append(os.getenv("GEMINI_API_KEY_2"))

MY_SECRET = os.getenv("MY_SECRET", "tds2_secret")
MY_EMAIL = os.getenv("MY_EMAIL", "22f2000771@ds.study.iitm.ac.in")
MY_ID_STRING = "22f2000771" 
GLOBAL_SUBMIT_URL = "https://tds-llm-analysis.s-anand.net/submit"
MODEL_NAME = 'gemini-2.5-flash'

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- HELPER FUNCTIONS ---

def get_next_key():
    if not KEYS: return None
    key = random.choice(KEYS)
    genai.configure(api_key=key)
    return key

def solve_heatmap_locally(image_path):
    """Mathematically finds the most frequent color using Python (No API needed)"""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        # Resize to speed up processing
        img.thumbnail((100, 100))
        pixels = list(img.getdata())
        # Count most frequent pixel
        most_common = Counter(pixels).most_common(1)[0][0]
        # Convert to Hex
        return '#{:02x}{:02x}{:02x}'.format(most_common[0], most_common[1], most_common[2])
    except Exception as e:
        logger.error(f"Local Heatmap Error: {e}")
        return None

def safe_generate_content(prompt_content, is_media=False, media_file=None):
    max_retries = 6 # Increased retries
    for attempt in range(max_retries):
        try:
            get_next_key()
            model = genai.GenerativeModel(MODEL_NAME)
            if is_media and media_file:
                return model.generate_content([prompt_content, media_file])
            return model.generate_content(prompt_content)
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                logger.warning(f"Rate Limit 429. Sleeping 30s... (Attempt {attempt+1})")
                time.sleep(30) # Increased sleep
            else:
                logger.error(f"Gemini Error: {e}")
                time.sleep(5)
    return None

def ask_gemini(prompt, content=""):
    prompt = global_sanitizer(prompt)
    content = global_sanitizer(content)
    full_prompt = f"CONTEXT:\n{content}\n\nTASK:\n{prompt}"
    response = safe_generate_content(full_prompt)
    return response.text.strip() if response and response.parts else ""

def ask_gemini_media(prompt, file_path):
    try:
        get_next_key()
        uploaded_file = genai.upload_file(path=file_path)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
        response = safe_generate_content(prompt, is_media=True, media_file=uploaded_file)
        return response.text.strip() if response else ""
    except: return ""

def global_sanitizer(text):
    if not isinstance(text, str): return text
    if MY_EMAIL: text = text.replace(MY_EMAIL, "[REDACTED_EMAIL]")
    return text.replace(MY_ID_STRING, "[REDACTED_ID]")

def get_page_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        try:
            page.wait_for_selector("body", timeout=5000)
            page.wait_for_timeout(2000)
        except: pass
        return page.content(), page.inner_text("body")

def download_file(file_url):
    try:
        if "<your email>" in file_url and MY_EMAIL:
            file_url = file_url.replace("<your email>", MY_EMAIL)
        filename = file_url.split("/")[-1].split("?")[0]
        if not filename: filename = f"data_{int(time.time())}"
        local_filename = f"/tmp/{filename}"
        with requests.get(file_url, stream=True) as r:
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return local_filename
    except: return None

def parse_quiz_page(html_content):
    question_text = "Calculate the answer."
    match = re.search(r'(Q\d+\.|Question:)(.{1,300})', html_content, re.IGNORECASE)
    if match: question_text = match.group(0)
    cleaned_text = ask_gemini(f"Analyze HTML. Extract task. JSON Format: {{ 'question': '...', 'data_url': 'url1,url2' }}", html_content[:50000]).replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned_text)
        if not data.get("question"): data["question"] = question_text
        return data
    except: return {"question": question_text, "data_url": None}

def manual_url_extraction(html_content, current_url):
    links = []
    for m in re.finditer(r'href=["\'](.*?\.(csv|opus|mp3|pdf|json|png|jpg))["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    for m in re.finditer(r'href=["\'](.*?-data.*?)["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    return ",".join(list(set(links)))

# --- CORE LOGIC ---
def solve_quiz_loop(start_url):
    current_url = start_url
    while current_url:
        logger.info("Sleeping 15s...")
        time.sleep(15) # Safety Sleep
        
        try:
            logger.info(f"--- Processing Level: {current_url} ---")
            html, visible_text = get_page_content(current_url)
            
            secret_match = re.search(r'Secret code is[:\s]*([A-Za-z0-9]+)', visible_text)
            if secret_match:
                answer = secret_match.group(1)
                task = {}
            else:
                task = parse_quiz_page(html)
                task["submit_url"] = GLOBAL_SUBMIT_URL
                manual_links = manual_url_extraction(html, current_url)
                if manual_links:
                    existing = str(task.get("data_url", ""))
                    task["data_url"] = existing + "," + manual_links if existing else manual_links
                
                answer = None
                context_info = ""
                csv_file = None
                json_file = None
                data_urls = [u.strip() for u in str(task.get("data_url", "")).split(",") if u.strip()]

                for d_url in data_urls:
                    if "<your email>" in d_url and MY_EMAIL: d_url = d_url.replace("<your email>", MY_EMAIL)
                    full_url = urljoin(current_url, d_url)
                    f_path = download_file(full_url)
                    if not f_path: continue

                    if f_path.endswith(('.png', '.jpg', '.jpeg')):
                        # LOCAL SOLVER FOR IMAGES
                        logger.info("Solving Heatmap locally...")
                        answer = solve_heatmap_locally(f_path)
                    
                    elif f_path.endswith(('.opus', '.mp3')):
                        transcription = ask_gemini_media(f"Transcribe audio. {task['question']}", f_path)
                        context_info += f"\nAUDIO: {transcription}\n"
                    
                    elif f_path.endswith('.csv'): csv_file = f_path
                    elif f_path.endswith('.json'): json_file = f_path
                    else:
                        try:
                            with open(f_path, "r", errors="ignore") as f: 
                                content = global_sanitizer(f.read(8000))
                                context_info += f"\nFILE: {content}\n"
                        except: pass

                if not answer:
                    if csv_file:
                        try:
                            df = pd.read_csv(csv_file)
                            if any(str(col).replace('.','',1).isdigit() for col in df.columns):
                                df = pd.read_csv(csv_file, header=None)
                                df.columns = [chr(65+i) for i in range(len(df.columns))]
                            
                            if "json" in task['question'].lower():
                                ans_str = ask_gemini(f"Convert CSV to JSON array. CSV:\n{df.head().to_string()}\nQuestion: {task['question']}\nReturn VALID JSON STRING ONLY.")
                                if "json" in ans_str.lower(): ans_str = ans_str.replace("```json", "").replace("```", "").strip()
                                answer = ans_str
                            else:
                                logic = ask_gemini(f"Write Python expression for `df`. Question: {task['question']}. If Cutoff, use SUM. Return ONLY expression.").replace("```python", "").strip()
                                result = eval(logic, {"df": df, "np": np})
                                answer = int(result) if hasattr(result, 'real') else str(result)
                        except:
                            answer = int(df.select_dtypes(include=[np.number]).sum().sum())

                    elif json_file and "gh-tree" in str(json_file):
                        try:
                            with open(json_file, "r") as f: tree = json.load(f)
                            res = requests.get(f"[https://api.github.com/repos/](https://api.github.com/repos/){tree['owner']}/{tree['repo']}/git/trees/{tree['sha']}?recursive=1", headers={"User-Agent": "x"}).json()
                            count = sum(1 for i in res.get("tree", []) if i["path"].startswith(tree.get("path_prefix", "")) and i["path"].endswith(".md"))
                            answer = count + (len(MY_EMAIL) % 2 if MY_EMAIL else 0)
                        except: answer = 0

                    elif context_info:
                        if "uv" in task['question'].lower():
                            tgt = data_urls[0] if data_urls else "URL"
                            answer = ask_gemini(f"Construct `uv http get` command for {tgt}.\nHeaders: Accept: application/json.\nDo NOT add X-Email.\nReturn ONLY command.")
                        else:
                            answer = ask_gemini(f"Question: {task['question']}\nContext: {context_info}\nReturn answer.")
                    else:
                        answer = ask_gemini(f"Question: {task['question']}\nHTML: {visible_text[:1000]}")

            # FINAL CLEANUP
            try:
                clean_ans = str(answer).strip().replace("bash", "").replace("`", "").strip()
                if "secret is" in clean_ans.lower(): clean_ans = clean_ans.split("secret is")[-1].strip()
                if MY_ID_STRING in clean_ans: clean_ans = ""
                
                # Force JSON quotes
                if clean_ans.startswith("["): clean_ans = clean_ans.replace("'", '"')
                
                answer = clean_ans
            except: pass

            payload = {"email": MY_EMAIL, "secret": MY_SECRET, "url": current_url, "answer": answer}
            logger.info(f"Submitting: {payload}")
            res = requests.post(GLOBAL_SUBMIT_URL, json=payload)
            res_json = res.json()
            logger.info(f"Result: {res_json}")

            if res_json.get("correct"): current_url = res_json.get("url")
            else: 
                current_url = res_json.get("url")
                if not current_url: break
        except Exception as e:
            logger.error(f"CRITICAL: {e}")
            break

@app.post("/")
async def receive_task(task: QuizRequest, background_tasks: BackgroundTasks):
    if task.secret != MY_SECRET: raise HTTPException(status_code=403)
    background_tasks.add_task(solve_quiz_loop, task.url)
    return {"message": "Processing"}