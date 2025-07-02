from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

result = llm.invoke("tell me about bangalore's history")
print(result)
