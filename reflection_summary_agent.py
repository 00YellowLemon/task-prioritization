from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Dict
import os
from dotenv import load_dotenv
import json
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough

load_dotenv()

# Ensure GOOGLE_API_KEY is set (you might want to handle this in main.py)
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the Gemini model (consider initializing in main.py and passing as dependency)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define Pydantic model for the structured output
class ReflectionSummaryOutput(BaseModel):
    reflection_summary: str = Field(..., description="One-line summary of the reflection.")

# Set up output parser for the first summary
# Update: Pass the entire ReflectionSummaryOutput model instead of its field type.
parser_reflection = PydanticOutputParser(pydantic_object=ReflectionSummaryOutput)

# Reflection Summary Logic with structured output
reflection_summary_prompt_template_structured = PromptTemplate(
    input_variables=["reflection"],
    template="""Please provide a one-line summary of the following reflection and format your response as a JSON object with the key 'reflection_summary':

{reflection}

{format_instructions}""",
    partial_variables={"format_instructions": parser_reflection.get_format_instructions()}
)

reflection_summary_chain_structured = RunnableSequence(
    {"reflection": RunnablePassthrough()},
    reflection_summary_prompt_template_structured,
    model,
    parser_reflection
)

# Define a Runnable that performs only the first summary
create_reflection_summary_agent = RunnableSequence(
    {"reflection_data": RunnablePassthrough()},
    RunnableLambda(lambda input_data: {"reflection": "\n".join([f"Question: {q}\nAnswer: {a}" 
                                                               for q, a in input_data['reflection_data'].items()])}),
    reflection_summary_chain_structured,
    RunnableLambda(lambda x: ReflectionSummaryOutput(reflection_summary=x))
)
