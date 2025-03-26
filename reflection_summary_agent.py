from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Output model (strict 12-word summary)
class ReflectionSummary(BaseModel):
    summary: str = Field(..., description="One-line summary of the entire reflection (max 12 words)")

# Create chain components
summary_parser = PydanticOutputParser(pydantic_object=ReflectionSummary)

prompt = PromptTemplate(
    template="""Summarize this reflection in one line (max 12 words):
    
    reflection: {input}
    
    Respond in this exact JSON format:
    {{
        "summary": "your_concise_summary_here"
    }}
    Focus on the big picture, not insights. Keep it short.""",
    input_variables=["input"],
    partial_variables={"format_instructions": summary_parser.get_format_instructions()}
)

# Chain
summary_chain = (
    {"input": RunnablePassthrough()} 
    | prompt 
    | model 
    | summary_parser
)