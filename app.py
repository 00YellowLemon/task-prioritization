from typing import Dict, Optional, Type
import os
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langserve import add_routes
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage
import uvicorn
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# Load environment variables from .env file

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Set it before running the server.")

# Task Prioritization Logic (remains the same)
class TaskInput(BaseModel):
    task: str = Field(description="The task to prioritize")

class TaskPrioritization(BaseModel):
    importance: bool = Field(description="Whether the task is important (True/False)")
    urgency: bool = Field(description="Whether the task is urgent (True/False)")

response_schemas = [
    ResponseSchema(name="importance", description="Whether the task is important. Must be 'true' or 'false'"),
    ResponseSchema(name="urgency", description="Whether the task is urgent. Must be 'true' or 'false'")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

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

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

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

# Simplified Reflection Insight Logic (using LLMChain directly)
class ReflectionInsightInput(BaseModel):
    reflection: str = Field(..., description="User's reflection with questions and answers.")

insight_template = """
You are an insightful reflection analyst. Your job is to provide deep insights based on a user's reflection,
which includes questions they've been asked and their answers.

When analyzing reflections, consider:
1. Patterns in the user's thinking
2. Potential blind spots or biases
3. Areas for further exploration or growth
4. Connections between different answers
5. Underlying themes or values revealed

Provide thoughtful, nuanced insights that help the user gain new perspectives on their reflections.

User Reflection:
{reflection}
"""

insight_prompt = PromptTemplate(
    template=insight_template,
    input_variables=["reflection"]
)

insight_chain = LLMChain(llm=model, prompt=insight_prompt)

def get_insights(reflection: str) -> str:
    response = insight_chain.invoke({"reflection": reflection})
    return response["text"]

# FastAPI App
app = FastAPI(
    title="Task Prioritization and Reflection Insights API",
    description="API for prioritizing tasks and generating reflection insights using Gemini",
    version="1.0.0",
)

add_routes(
    app,
    prioritize_task_chain(),
    path="/prioritize-task",
    input_type=TaskInput,
    output_type=TaskPrioritization,
)

@app.post("/get-reflection-insights/")
async def reflection_insights(reflection_input: ReflectionInsightInput) -> Dict[str, str]:
    insights = get_insights(reflection_input.reflection)
    return {"insights": insights}

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Task Prioritization and Reflection Insights API is running!",
        "documentation": "/docs",
        "endpoints": {"prioritize_task": "/prioritize-task/invoke", "reflection_insights": "/get-reflection-insights"}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    