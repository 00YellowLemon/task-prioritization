import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the chains
from task_prioritization_agent import chain as task_chain
from reflection_insights_agent import reflection_insights_chain
from reflection_summary_agent import summary_chain

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Task Prioritization API",
    description="API for task prioritization and reflection analysis using AI"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request bodies
class TaskRequest(BaseModel):
    task: str

class ReflectionInsightsRequest(BaseModel):
    reflections: str

class ReflectionSummaryRequest(BaseModel):
    reflection: str

# Task prioritization endpoint
@app.post("/task")
async def prioritize_task(request: TaskRequest):
    try:
        result = task_chain.invoke({"task": request.task})
        return JSONResponse(content=result.dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Reflection insights endpoint
@app.post("/reflection_insights")
async def get_reflection_insights(request: ReflectionInsightsRequest):
    try:
        result = reflection_insights_chain.invoke(request.reflections)
        return JSONResponse(content=result.dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Reflection summary endpoint
@app.post("/reflection_summary")
async def get_reflection_summary(request: ReflectionSummaryRequest):
    try:
        result = summary_chain.invoke(request.reflection)
        return JSONResponse(content=result.dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8800)))
