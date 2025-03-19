# app.py
from typing import Dict
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langserve import add_routes
import os

# Check if Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY environment variable not set. Set it before running the server.")

# Define the input schema
class TaskInput(BaseModel):
    task: str = Field(description="The task to prioritize")

# Define the output schema
class TaskPrioritization(BaseModel):
    importance: bool = Field(description="Whether the task is important (True/False)")
    urgency: bool = Field(description="Whether the task is urgent (True/False)")

# Define the response schemas for the LLM
response_schemas = [
    ResponseSchema(name="importance", description="Whether the task is important. Must be either 'true' or 'false'"),
    ResponseSchema(name="urgency", description="Whether the task is urgent. Must be either 'true' or 'false'")
]

# Create the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get the format instructions
format_instructions = output_parser.get_format_instructions()

# Create the prompt template
template = """
You are a task prioritization assistant. Your job is to analyze the given task and determine its importance and urgency.

Task: {task}

Importance means the task has significant value or impact on goals, objectives, or outcomes.
Urgency means the task requires immediate attention or action due to time constraints.

Please provide your assessment in the following format:
{format_instructions}
"""

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(template)
    ],
    input_variables=["task"],
    partial_variables={"format_instructions": format_instructions}
)

# Create the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Create the task prioritization chain
def prioritize_task_chain():
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.runnable import RunnableSequence
    def parse_output(output):
        parsed = output_parser.parse(output)
        return {
            "importance": parsed["importance"].lower() == "true",
            "urgency": parsed["urgency"].lower() == "true"
        }
    return (
        {"task": RunnablePassthrough()}
        | prompt
        | model
        | output_parser.parse
        | RunnablePassthrough.assign(importance = lambda x : x["importance"].lower() == "true", urgency = lambda x: x["urgency"].lower() == "true")
    )
# Create FastAPI app
app = FastAPI(
    title="Task Prioritization API",
    description="A simple API for prioritizing tasks based on importance and urgency using Gemini",
    version="1.0.0",
)

# Add routes for the task prioritization chain
add_routes(
    app,
    prioritize_task_chain(),
    path="/prioritize-task",
    input_type=TaskInput,
    output_type=TaskPrioritization,
)

# Add documentation route
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Task Prioritization API is running!",
        "documentation": "/docs",
        "endpoints": {
            "prioritize_task": "/prioritize-task/invoke"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)