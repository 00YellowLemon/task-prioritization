from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.pydantic_v1 import BaseModel, Field
import os
from dotenv import load_dotenv
import json  # Import the json module

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
You are an insightful reflection analyst. Your job is to provide between 1 and 4 distinct insights based on the user's reflection. For each insight, provide analysis under a consistent set of categories, with a maximum of 6 categories per insight.

The categories you should consider using for each insight are:

**Category Options:**
- Patterns in Thinking
- Potential Blind Spots or Biases
- Areas for Further Exploration or Growth
- Connections Between Answers
- Underlying Themes or Values
- Actionable Insights & Next Steps
- Strengths Identified
- Potential Challenges or Obstacles
- Key Learning

Please structure your response as follows:

**Insight 1:**
  **Category 1:** [Analysis for Category 1]
  **Category 2:** [Analysis for Category 2]
  ... (up to 6 categories)

**Insight 2:**
  **Category 1:** [Analysis for Category 1]
  **Category 2:** [Analysis for Category 2]
  ... (up to 6 categories, same as Insight 1)

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
        category_lines = block.strip().split("\n  **Category ")
        categories = {}
        for category_line in category_lines[1:]:  # Skip the first part before the first category
            if ":" in category_line:
                category_name_part, category_content = category_line.split(":", 1)
                category_name = category_name_part.strip()
                category_content = category_content.strip()
                categories[category_name] = category_content
        insight.update(categories)
        insights_data.append(insight)

    return json.dumps(insights_data, indent=2)