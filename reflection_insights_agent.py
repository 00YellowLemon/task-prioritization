import os
from dotenv import load_dotenv
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load environment variables
load_dotenv()

# Pydantic Models for Structured Output
class Insight(BaseModel):
    thinking: str = Field(..., description="Analysis of your thought patterns")
    blindspot: str = Field(..., description="What you might be overlooking")
    growth: str = Field(..., description="Your growth opportunity")
    action: str = Field(..., description="Your recommended next step")

class Insights(BaseModel):
    insights: List[Insight]

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create Output Parser
parser = PydanticOutputParser(pydantic_object=Insights)

# Define Prompt Template
insight_template = """
You're my personal reflection coach. I'm sharing these thoughts with you:

{reflections}

For each entry I've shared, please help me understand:
1. What my thinking patterns reveal about me
2. What perspectives I might be missing
3. Where I have the most potential to grow
4. One concrete action I could take

Format your response as clean JSON with these exact fields for each insight:
- "thinking"
- "blindspot" 
- "growth"
- "action"

Structure the output like this:
{{
  "insights": [
    {{
      "thinking": "...",
      "blindspot": "...", 
      "growth": "...",
      "action": "..."
    }}
  ]
}}
"""

# Create Prompt Template
insight_prompt = PromptTemplate(
    template=insight_template,
    input_variables=["reflections"]
)

# Create the final chain
reflection_insights_chain = (
    RunnableParallel({"reflections": RunnablePassthrough()})
    | insight_prompt 
    | model 
    | parser
)