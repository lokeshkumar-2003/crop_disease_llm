import json

def load_remedies():
    with open("data/remedies.json", "r", encoding="utf-8") as f:
        return json.load(f)

def adjust_dosage(base_dosage: float, humidity: int, rainfall: str) -> float:
    adjusted = base_dosage

    if humidity >= 75:
        adjusted *= 1.10

    if rainfall in ["moderate rain", "heavy rain"]:
        adjusted *= 1.15

    return round(adjusted, 2)
