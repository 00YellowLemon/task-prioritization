from typing import Dict
import os
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langserve import add_routes
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage
import uvicorn

from dotenv import load_dotenv
load_dotenv()


# Load environment variables from .env file

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Set it before running the server.")

# Define the input schema
class TaskInput(BaseModel):
    task: str = Field(description="The task to prioritize")

# Define the output schema
class TaskPrioritization(BaseModel):
    importance: bool = Field(description="Whether the task is important (True/False)")
    urgency: bool = Field(description="Whether the task is urgent (True/False)")

# Define response schemas
response_schemas = [
    ResponseSchema(name="importance", description="Whether the task is important. Must be 'true' or 'false'"),
    ResponseSchema(name="urgency", description="Whether the task is urgent. Must be 'true' or 'false'")
]

# Create the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Create the prompt template
def get_task_prioritization_prompt_template():
    template = (
        """
        You are a task prioritization assistant. Analyze the given task and determine its importance and urgency.
        Task: {task}
        
        Importance means the task has significant value or impact on goals.
        Urgency means the task requires immediate attention.
        
        Provide your assessment in JSON format:
        {format_instructions}
        """
    )
    return ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(template)],
        input_variables=["task"],
        partial_variables={"format_instructions": format_instructions}
    )

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Define the prioritization chain
def prioritize_task_chain():
    def extract_content(message: AIMessage) -> str:
        return message.content if hasattr(message, 'content') else ""

    def parse_output(output: str):
        parsed = output_parser.parse(output)
        return {
            "importance": parsed.get("importance", "false").lower() == "true",
            "urgency": parsed.get("urgency", "false").lower() == "true"
        }

    return (
        {"task": RunnablePassthrough()} 
        | get_task_prioritization_prompt_template()
        | model 
        | RunnableLambda(extract_content) 
        | RunnableLambda(parse_output)
    )

# Create FastAPI app
app = FastAPI(
    title="Task Prioritization API",
    description="API for prioritizing tasks based on importance and urgency using Gemini",
    version="1.0.0",
)

# Add API routes
add_routes(
    app,
    prioritize_task_chain(),
    path="/prioritize-task",
    input_type=TaskInput,
    output_type=TaskPrioritization,
)

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Task Prioritization API is running!",
        "documentation": "/docs",
        "endpoints": {"prioritize_task": "/prioritize-task/invoke"}
    }

# Start the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
