from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import quiz
import uvicorn
import os
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz.quiz_router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use 8000 locally, dynamic in Railway
    uvicorn.run("main:app", host="0.0.0.0", port=port)
