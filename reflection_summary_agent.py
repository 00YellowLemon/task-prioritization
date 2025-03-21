from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.pydantic_v1 import BaseModel, Field
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Ensure GOOGLE_API_KEY is set (you might want to handle this in main.py)
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the Gemini model (consider initializing in main.py and passing as dependency)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Simplified Reflection Insight Logic
class ReflectionInsightInput(BaseModel):
    reflection: str = Field(..., description="User's reflection with questions and answers.")

insight_template = """
You are an insightful reflection analyst. Your job is to provide between 1 and 4 distinct insights based on the user's reflection. For each insight, provide analysis under the following four consistent categories:

**Thinking:** (Describe any recurring patterns or approaches in the user's responses)
**Blindspot:** (Identify any potential areas where the user might have overlooked something or shown bias)
**Growth:** (Suggest areas for further exploration or development for the user)
**Action:** (Provide a concise actionable insight or next step for the user)

Please structure your response as follows:

**Insight 1:**
  **Thinking:** [Analysis for Thinking]
  **Blindspot:** [Analysis for Blindspot]
  **Growth:** [Analysis for Growth]
  **Action:** [Analysis for Action]

**Insight 2:**
  **Thinking:** [Analysis for Thinking]
  **Blindspot:** [Analysis for Blindspot]
  **Growth:** [Analysis for Growth]
  **Action:** [Analysis for Action]

... (up to 4 insights in total)

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
    raw_text = response["text"]

    insights_data = []
    insight_blocks = raw_text.strip().split("\n\n**Insight ")

    for i, block in enumerate(insight_blocks[1:]):  # Skip the first empty element
        insight_number = i + 1
        insight = {"Insight": insight_number}
        category_lines = block.strip().split("\n  **")
        categories = {}
        current_category = None
        for line in category_lines[1:]:
            if ":" in line:
                category_name_part, category_content = line.split(":", 1)
                category_name = category_name_part.strip()
                category_content = category_content.strip()
                categories[category_name.lower()] = category_content # Convert to lowercase for consistency
                current_category = category_name.lower()
            elif current_category and line.strip():
                # Handle potential multi-line content within a category
                categories[current_category] += "\n" + line.strip()
        insight.update(categories)
        insights_data.append(insight)

    return json.dumps(insights_data, indent=2)