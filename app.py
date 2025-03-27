import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

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

# Task prioritization endpoint
@app.post("/task")
async def prioritize_task(task: str):
    try:
        result = task_chain.invoke({"task": task})
        return JSONResponse(content=result.dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Reflection insights endpoint
@app.post("/reflection_insights")
async def get_reflection_insights(reflections: str):
    try:
        result = reflection_insights_chain.invoke(reflections)
        return JSONResponse(content=result.dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Reflection summary endpoint
@app.post("/reflection_summary")
async def get_reflection_summary(reflection: str):
    try:
        result = summary_chain.invoke(reflection)
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
