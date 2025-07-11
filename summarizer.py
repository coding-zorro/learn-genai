# Import necessary libraries
from transformers import pipeline  # For using pre-trained models from Hugging Face
import gradio as gr  # For creating a simple web UI

# Initialize the summarization pipeline using a pre-trained model from Hugging Face.
# The "facebook/bart-large-cnn" model is specifically fine-tuned for summarization tasks.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Define the function that will take input text and return its summary.
# This function is called by the Gradio interface.
def summarize_text(input_text):
    # Use the summarizer pipeline to generate the summary.
    # `max_length` and `min_length` control the length of the summary in tokens.
    # `do_sample=False` ensures deterministic output.
    result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
    # The result is a list of dictionaries, so we extract the summary text from the first element.
    return result[0]["summary_text"]


# Create and launch a Gradio web interface.
# `fn` is the function to be executed.
# `inputs` and `outputs` define the type of UI components (in this case, text boxes).
# `title` sets the title of the web page.
# The interface will be available at a local URL (e.g., http://127.0.0.1:7860).
gr.Interface(
    fn=summarize_text, inputs="textbox", outputs="textbox", title="Simple Summarizer"
).launch()
