from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    location: str


parser = JsonOutputParser(pydantic_object=Person)

prompt = PromptTemplate.from_template(
    """
        You are an information extractor AI. Given the following input text, extract the person's name, age and location.

        Respond ONLY with the result in **valid JSON** format — no commentary, no explanation, no markdown.
        Respond only in the following JSON format:

        {format_instructions}

        Text:
        {text}
    """
)

llm = OllamaLLM(model="mistral")
chain = prompt | llm | parser
input = {
    "format_instructions": parser.get_format_instructions,
    "text": "John Doe is a 34-year-old software engineer. His is from Los Angeles",
}
result = chain.invoke(input)
print(result)
