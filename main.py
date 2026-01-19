


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from bson import ObjectId
from bson.errors import InvalidId
from pydantic import BaseModel
from typing import List
import os
import httpx
from dotenv import load_dotenv
from database import db

# ======================================
# Load environment variables
# ======================================
load_dotenv()

# ======================================
# Initialize FastAPI app with Swagger docs enabled
# ======================================
app = FastAPI(
    title="Gemini Backend API",
    description="FastAPI backend for AI Interview Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ======================================
# Middleware (CORS)
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://your-frontend.vercel.app"  # production frontend

    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ======================================
# Environment Config
# ======================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY in environment variables")

# ======================================
# Models
# ======================================
class Message(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "gpt-4o"
    max_tokens: int = 1000
    messages: List[Message]

class Question(BaseModel):
    questionText: str

class JobResponse(BaseModel):
    id: str
    title: str
    description: str
    questions: List[Question]
    numberOfQuestions: int

# ======================================
# Health Checks
# ======================================
@app.get("/", summary="Root health check")
@app.get("/health", summary="Health check endpoint", tags=["Monitoring"])
def health_check():
    return JSONResponse(
        content={"status": "MongoDB connected successfully"},
        status_code=200
    )

# ======================================
# OpenAI Proxy Endpoint
# ======================================
@app.post("/api/openai", summary="Proxy to OpenAI API")
async def openai_proxy(request: OpenAIRequest):
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": 0.7
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENAI_API_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {response.text}"
            )

        return response.json()

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to OpenAI API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ======================================
# Helper Function to Convert Mongo ObjectIDs
# ======================================
def convert_objectid(data):
    if isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_objectid(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

# ======================================
# Job Details Endpoint
# ======================================
@app.get("/jobs/{_id}", summary="Fetch Job Details by ID")
async def get_job_details(_id: str):
    try:
        object_id = ObjectId(_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid job ID")

    job = await db.jobs.find_one({"_id": object_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job = convert_objectid(job)

    return {
        "job_id": job["_id"],
        "title": job.get("jobTitle"),
        "description": job.get("plainTextJobDescription"),
        "questions": job.get("questions", [])
    }
