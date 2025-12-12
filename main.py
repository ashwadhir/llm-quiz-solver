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
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LOAD ALL KEYS
KEYS = []
if os.getenv("GEMINI_API_KEY"): KEYS.append(os.getenv("GEMINI_API_KEY"))
if os.getenv("GEMINI_API_KEY_2"): KEYS.append(os.getenv("GEMINI_API_KEY_2"))
# Add more if you have them: GEMINI_API_KEY_3, etc.

MY_SECRET = os.getenv("MY_SECRET", "tds2_secret")
MY_EMAIL = os.getenv("MY_EMAIL", "22f2000771@ds.study.iitm.ac.in")
MY_ID_STRING = "22f2000771" 
GLOBAL_SUBMIT_URL = "https://tds-llm-analysis.s-anand.net/submit"
MODEL_NAME = 'gemini-2.5-flash' # Using your preferred model

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# --- HELPER FUNCTIONS ---

def get_next_key():
    """Rotates keys to avoid Rate Limits"""
    if not KEYS: return None
    key = random.choice(KEYS) # Random rotation is simple and effective
    genai.configure(api_key=key)
    return key

def get_page_content(url):
    logger.info(f"Scraping: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        try:
            page.wait_for_selector("body", timeout=5000)
            page.wait_for_timeout(2000)
        except: pass
        content = page.content()
        visible_text = page.inner_text("body")
        browser.close()
        return content, visible_text

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

def global_sanitizer(text_input):
    if not isinstance(text_input, str): return text_input
    text = text_input
    if MY_EMAIL: text = text.replace(MY_EMAIL, "[REDACTED_EMAIL]")
    text = text.replace(MY_ID_STRING, "[REDACTED_ID]")
    return text

def safe_generate_content(prompt_content, is_media=False, media_file=None):
    """Smart Retry with Key Rotation"""
    max_retries = len(KEYS) * 2 # Try every key twice
    
    for attempt in range(max_retries):
        try:
            get_next_key() # Rotate key
            model = genai.GenerativeModel(MODEL_NAME)
            
            if is_media and media_file:
                return model.generate_content([prompt_content, media_file])
            else:
                return model.generate_content(prompt_content)
                
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Quota" in err_str:
                logger.warning(f"Rate Limit Hit. Switching Key... (Attempt {attempt+1})")
                time.sleep(2) # Short sleep to let the switch happen cleanly
            else:
                logger.error(f"Gemini Error: {e}")
                return None
    return None

def ask_gemini(prompt, content=""):
    prompt = global_sanitizer(prompt)
    content = global_sanitizer(content)
    full_prompt = f"CONTEXT:\n{content}\n\nTASK:\n{prompt}"
    
    response = safe_generate_content(full_prompt)
    return response.text.strip() if response and response.parts else ""

def ask_gemini_media(prompt, file_path):
    logger.info(f"Processing Media: {file_path}")
    try:
        # File upload needs a key too, so we set it first
        get_next_key() 
        uploaded_file = genai.upload_file(path=file_path)
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        response = safe_generate_content(prompt, is_media=True, media_file=uploaded_file)
        return response.text.strip() if response else ""
    except Exception as e:
        logger.error(f"Media Error: {e}")
        return ""

def parse_quiz_page(html_content):
    question_text = "Calculate the answer."
    match = re.search(r'(Q\d+\.|Question:)(.{1,300})', html_content, re.IGNORECASE)
    if match: question_text = match.group(0)

    prompt = f"""
    Analyze HTML. Extract task.
    JSON Format: {{ "question": "...", "data_url": "url1,url2" }}
    """
    cleaned_text = ask_gemini(prompt, html_content[:50000]) 
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned_text)
        if not data.get("question"): data["question"] = question_text
        return data
    except:
        return {"question": question_text, "data_url": None}

def manual_url_extraction(html_content, current_url):
    links = []
    for m in re.finditer(r'href=["\'](.*?\.(csv|opus|mp3|pdf|json|png|jpg))["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    for m in re.finditer(r'href=["\'](.*?-data.*?)["\']', html_content):
        links.append(urljoin(current_url, m.group(1)))
    return ",".join(list(set(links)))

# --- CORE SOLVER LOGIC ---

def solve_quiz_loop(start_url):
    current_url = start_url
    
    while current_url:
        # NO SLEEP NEEDED - Keys will rotate automatically
        
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

                task["question"] = global_sanitizer(task["question"])
                logger.info(f"Task Parsed: {task}")
                
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

                    if f_path.endswith(('.opus', '.mp3', '.wav')):
                        transcription = ask_gemini_media(f"Transcribe. {task['question']}", f_path)
                        context_info += f"\nAUDIO TRANSCRIPT: {transcription}\n"
                    
                    elif f_path.endswith(('.png', '.jpg', '.jpeg')):
                        desc = ask_gemini_media(f"Analyze image. {task['question']}. Return ONLY the requested value (e.g. hex color).", f_path)
                        context_info += f"\nIMAGE ANALYSIS: {desc}\n"
                        if len(desc) < 20: answer = desc

                    elif f_path.endswith('.csv'): csv_file = f_path
                    elif f_path.endswith('.json'): json_file = f_path
                    else:
                        try:
                            with open(f_path, "r", errors="ignore") as f: content = f.read(8000)
                            content = global_sanitizer(content)
                            context_info += f"\nFILE CONTENT: {content}\n"
                        except: pass

                if not answer:
                    if csv_file:
                        try:
                            df = pd.read_csv(csv_file)
                            if any(str(col).replace('.','',1).isdigit() for col in df.columns):
                                df = pd.read_csv(csv_file, header=None)
                                df.columns = [chr(65+i) for i in range(len(df.columns))]
                            
                            preview = df.head().to_string()
                            
                            if "json" in task['question'].lower() or "array" in task['question'].lower():
                                ans_str = ask_gemini(f"Convert CSV to JSON. CSV:\n{preview}\nQuestion: {task['question']}\nReturn VALID JSON STRING only.")
                                if "json" in ans_str.lower(): ans_str = ans_str.replace("```json", "").replace("```", "").strip()
                                answer = ans_str
                            else:
                                logic = ask_gemini(f"Write Python expression for `df` to solve: {task['question']}. Return ONLY expression.").replace("```python", "").strip()
                                result = eval(logic, {"df": df, "np": np})
                                answer = int(result) if hasattr(result, 'real') else str(result)
                        except:
                            answer = int(df.select_dtypes(include=[np.number]).sum().sum())

                    elif json_file and "gh-tree" in str(json_file):
                        try:
                            with open(json_file, "r") as f: tree_cfg = json.load(f)
                            owner, repo, sha = tree_cfg["owner"], tree_cfg["repo"], tree_cfg["sha"]
                            api_url = f"[https://api.github.com/repos/](https://api.github.com/repos/){owner}/{repo}/git/trees/{sha}?recursive=1"
                            gh_res = requests.get(api_url, headers={"User-Agent": "agent"}).json()
                            count = sum(1 for i in gh_res.get("tree", []) if i["path"].startswith(tree_cfg.get("path_prefix", "")) and i["path"].endswith(".md"))
                            offset = len(MY_EMAIL) % 2 if MY_EMAIL else 0
                            answer = count + offset
                        except: answer = 0

                    elif context_info:
                        if "uv" in task['question'].lower():
                            tgt = data_urls[0] if data_urls else "MISSING"
                            answer = ask_gemini(f"Construct `uv http get` command for {tgt}.\nHeaders: Accept: application/json.\nDo NOT add X-Email unless asked.\nUse short flags (-H).\nReturn ONLY command.")
                        else:
                            answer = ask_gemini(f"Question: {task['question']}\nContext: {context_info}\nReturn answer.")
                    
                    else:
                        answer = ask_gemini(f"Question: {task['question']}\nHTML: {visible_text[:1000]}")

            try:
                clean_ans = str(answer).strip().replace("**", "").replace("`", "").replace('"', '').replace("'", "")
                
                if "/" in clean_ans and "<" in clean_ans:
                    clean_ans = clean_ans.split("<")[0].strip()

                if len(clean_ans) > 150 and not clean_ans.startswith("uv") and not clean_ans.startswith("["): 
                    code_match = re.search(r'\b([a-zA-Z0-9]{5,20})\b', clean_ans)
                    if code_match: clean_ans = code_match.group(1)

                if "secret is" in clean_ans.lower(): clean_ans = clean_ans.split("secret is")[-1].strip()
                if MY_ID_STRING in clean_ans: clean_ans = ""
                
                if not clean_ans.startswith("[") and not clean_ans.startswith("{"):
                    if clean_ans.replace('.','',1).isdigit():
                        answer = float(clean_ans) if '.' in clean_ans else int(clean_ans)
                    else:
                        answer = clean_ans
                else:
                    answer = clean_ans.replace("'", '"')
            except: pass

            payload = {"email": MY_EMAIL, "secret": MY_SECRET, "url": current_url, "answer": answer}
            logger.info(f"Submitting: {payload}")
            res = requests.post(GLOBAL_SUBMIT_URL, json=payload)
            try: res_json = res.json()
            except: res_json = {}
            logger.info(f"Result: {res_json}")

            if res_json.get("correct"):
                current_url = res_json.get("url")
            else:
                current_url = res_json.get("url")
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