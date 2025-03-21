from fastapi import FastAPI
from langserve import add_routes
from agents.task_prioritization_agent import prioritize_task_chain, TaskInput, TaskPrioritization
from agents.reflection_insights_agent import ReflectionInsightInput, get_insights
from agents.reflection_summary_agent import ReflectionSummaryInput, create_reflection_summary
from typing import Dict
import uvicorn
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Set it before running the server.")

# Initialize the Gemini model here to share across agents (optional, but good practice)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(
    title="Task Prioritization and Reflection Insights API",
    description="API for prioritizing tasks and generating reflection insights using Gemini",
    version="1.0.0",
)

# Add routes for the task prioritization agent
add_routes(
    app,
    prioritize_task_chain(),
    path="/prioritize-task",
    input_type=TaskInput,
    output_type=TaskPrioritization,
)

# Add route for reflection insights
@app.post("/get-reflection-insights/")
async def reflection_insights(reflection_input: ReflectionInsightInput) -> Dict[str, str]:
    insights = get_insights(reflection_input.reflection)
    return {"insights": insights}

# Add route for reflection summary
@app.post("/summarize-reflection/")
async def summarize_reflection(reflection_input: ReflectionSummaryInput) -> Dict[str, str]:
    reflection_summary, summary_summary = create_reflection_summary(reflection_input.reflection_data)
    return {"reflection_summary": reflection_summary, "summary_summary": summary_summary}

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Task Prioritization and Reflection Insights API is running!",
        "documentation": "/docs",
        "endpoints": {
            "prioritize_task": "/prioritize-task/invoke",
            "reflection_insights": "/get-reflection-insights",
            "summarize_reflection": "/summarize-reflection"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)