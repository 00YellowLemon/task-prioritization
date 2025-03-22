import os
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv

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



# Optional: Add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8800)))
