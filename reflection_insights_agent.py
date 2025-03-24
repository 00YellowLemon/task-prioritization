from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define Pydantic models for structured output
class Insight(BaseModel):
    thinking: str = Field(..., description="Analysis for Thinking")
    blindspot: str = Field(..., description="Analysis for Blindspot")
    growth: str = Field(..., description="Analysis for Growth")
    action: str = Field(..., description="Concise actionable insight or next step")

class Insights(BaseModel):
    insights: List[Dict[str, Insight]]

# Set up output parser
parser = PydanticOutputParser(pydantic_object=Insights)

insight_template_structured = """
You are an insightful reflection analyst. Your job is to provide between 1 and 4 distinct insights based on the user's reflection. For each insight, provide analysis under the following four consistent categories:

Thinking: (Describe any recurring patterns or approaches in the user's responses)
Blindspot: (Identify any potential areas where the user might have overlooked something or shown bias)
Growth: (Suggest areas for further exploration or development for the user)
Action: (Provide a concise actionable insight or next step for the user)

Please structure your response as a JSON array of objects, where each object represents an insight. Each insight object should have the following keys: "thinking", "blindspot", "growth", and "action".

Here are the format instructions:
{format_instructions}

User Reflection:
{reflection}
"""

insight_prompt_structured = PromptTemplate(
    template=insight_template_structured,
    input_variables=["reflection"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

reflection_insights_chain = RunnableSequence(
    {"reflection": RunnablePassthrough()},
    insight_prompt_structured,
    model,
    parser
)

def create_reflection_insights_chain():
    return reflection_insights_chain
