from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure GOOGLE_API_KEY is set (you might want to handle this in main.py)
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the Gemini model (consider initializing in main.py and passing as dependency)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Task Prioritization Logic
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