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

# Initialize the Gemini model
