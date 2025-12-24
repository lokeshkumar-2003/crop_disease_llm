import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-flash-latest")

def generate_advice(context):
    prompt = f"""
You are an agricultural advisory assistant.

Disease: {context['disease']}
Location: {context['location']}
Temperature: {context['temperature']} °C
Humidity: {context['humidity']} %
Rainfall: {context['rainfall']}

Chemical pesticide: {context['chemical']}
Dosage: {context['dosage']}

Organic remedies: {", ".join(context['organic'])}
Safety rules: {", ".join(context['safety'])}

Explain clearly in simple language for farmers.
"""

    response = model.generate_content(prompt)
    return response.text
