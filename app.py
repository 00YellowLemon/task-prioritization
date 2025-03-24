import os
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv

<<<<<<< HEAD
from reflection_insights_agent import get_insights as get_reflection_insights
from reflection_summary_agent import get_insights as get_reflection_summary
from task_prioritization_agent import prioritize_task_chain

load_dotenv()

# Load environment variables from .env file

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Set it before running the server.")

# Initialize the Gemini model

app = FastAPI()

# Add routes for the agents
add_routes(app, {
    "/reflection-insights": get_reflection_insights,
    "/reflection-summary": get_reflection_summary,
    "/task-prioritization": prioritize_task_chain()
})
=======
# Import the task prioritization chain
from task_prioritization_agent import chain
from reflection_insights_agent import reflection_insights_chain
from reflection_summary_agent import create_reflection_summary_agent

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Task Prioritization API",
    description="API for task prioritization using AI"
)

# Add route for task prioritization
add_routes(
    app, 
    chain,
    path="/task"
)

add_routes(
    app,
    reflection_insights_chain,
    path="/reflection_insights"
)

add_routes(
    app,
    create_reflection_summary_agent,
    path="/reflection_summary"

)

>>>>>>> breakdown-tasks


# Optional: Add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
<<<<<<< HEAD
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
=======
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8800)))
>>>>>>> breakdown-tasks
