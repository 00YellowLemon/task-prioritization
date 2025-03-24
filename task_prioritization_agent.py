from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Structured Output Model
class TaskPriority(BaseModel):
    """
    Represents the priority assessment of a task
    """
    is_important: bool = Field(
        description="Indicates whether the task is important",
        default=False
    )
    is_urgent: bool = Field(
        description="Indicates whether the task is urgent",
        default=False
    )


prompt_template = ChatPromptTemplate.from_template("""
    You are a task prioritization assistant. Analyze the given task carefully.

    Task: {task}

    Provide a precise assessment:
    - Determine if the task is IMPORTANT (True/False)
    - Determine if the task is URGENT (True/False)

    Strictly respond in JSON format:
    ```json
    {{
    "is_important": true/false,
    "is_urgent": true/false
    }}
""")

# Create the chain using the model's structured output method
chain = prompt_template | model.with_structured_output(TaskPriority)