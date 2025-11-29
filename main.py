from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

def process_quiz(task_url: str):
    print(f"PROCESSING QUIZ: {task_url}")
    # Scraping logic will go here

@app.post("/")
async def submit_task(request: QuizRequest, background_tasks: BackgroundTasks):
    # Verify Secret (Use environment variables in production, but hardcode for now is fine)
    MY_SECRET = "tds_project2_llm_quiz_solver" 
    
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    background_tasks.add_task(process_quiz, request.url)
    return {"message": "Task accepted"}

@app.get("/")
def home():
    return {"message": "Quiz Solver is running on Hugging Face!"}