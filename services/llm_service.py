from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LANGUAGE_INSTRUCTIONS = {
    "english":   "Respond ENTIRELY in English.",
    "tamil":     "Respond ENTIRELY in Tamil (தமிழ்) script only. Do NOT use English.",
    "hindi":     "Respond ENTIRELY in Hindi (हिंदी) script only. Do NOT use English.",
    "telugu":    "Respond ENTIRELY in Telugu (తెలుగు) script only. Do NOT use English.",
    "kannada":   "Respond ENTIRELY in Kannada (ಕನ್ನಡ) script only. Do NOT use English.",
    "malayalam": "Respond ENTIRELY in Malayalam (മലയാളം) script only. Do NOT use English.",
}

PROMPT_TEMPLATE = """You are an expert agronomist. Give structured, scannable advice to a farmer.

LANGUAGE RULE: {lang_instruction}
Translate ALL text including section headers, table labels, and bullets into the selected language.

OUTPUT — follow this EXACT structure:

🌾 **Crop Issue: {disease}**
_Location: {location} · Temp: {temperature}°C · Humidity: {humidity}%_
---
**🔬 What's Happening**
2 short sentences: what this disease is and why current weather worsens it.
---
**⚡ Immediate Action**
3 numbered steps to do RIGHT NOW. Specific, short.
1.
2.
3.
---
**💊 Chemical Treatment**
| Detail | Value |
|--------|-------|
| Chemical | {chemical} |
| Dose | {dosage_per_liter} per liter |
| Total Water | {water_required} L for {land_size_acre} acres |
| Best Time | Early morning or late evening |
| Repeat | After 7 days if needed |
---
**🌿 Organic Alternatives**
{organic_list}
One line per option explaining how it helps.
---
**⚠️ Safety Tips**
- Tip 1
- Tip 2
---
**✅ Expected Result**
One sentence: what the farmer sees in 5–7 days if done correctly.

RULES: Short sentences. Simple words. No long paragraphs. Be specific with numbers.
"""


def _build_prompt(context: dict) -> str:
    language = context.get("language", "english").lower()
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"])
    organic = context.get("organic", [])
    organic_list = "\n".join(f"- {o}" for o in organic[:4]) if organic else "- None specified"

    return PROMPT_TEMPLATE.format(
        lang_instruction=lang_instruction,
        disease=context.get("disease", "Unknown"),
        location=context.get("location", "Unknown"),
        temperature=context.get("temperature", "N/A"),
        humidity=context.get("humidity", "N/A"),
        land_size_acre=context.get("land_size_acre", "N/A"),
        chemical=context.get("chemical", "Unknown"),
        dosage_per_liter=context.get("dosage_per_liter", "N/A"),
        water_required=context.get("water_required", "N/A"),
        organic_list=organic_list,
    )


def generate_advice(context: dict) -> str:
    """Non-streaming: returns full advice string."""
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": _build_prompt(context)}],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=2048,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"🌾 **{context.get('disease', 'Crop Disease')} Detected**\n\n"
            f"Apply **{context.get('chemical', 'recommended pesticide')}** at "
            f"**{context.get('dosage_per_liter', 'standard dose')}** per liter.\n"
            f"Mix with **{context.get('water_required', 'required')} liters** of water.\n"
            f"Spray in early morning. Repeat after 7 days. Wear gloves.\n\n"
            f"_(Service error: {str(e)})_"
        )


def generate_advice_stream(context: dict):
    """
    Streaming version — yields token chunks from Groq.
    Use with FastAPI StreamingResponse(generate_advice_stream(ctx), media_type='text/plain')
    """
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": _build_prompt(context)}],
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=2048,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta