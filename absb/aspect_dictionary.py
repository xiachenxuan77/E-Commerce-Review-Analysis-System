"""
aspect_dictionary.py
Maintains all aspect-to-keyword mappings and saves them to JSON.
"""

import json
import os

ASPECT_DICTIONARY = {
    "Battery": [
        "battery", "charging", "charger", "charge", "battery life",
        "power", "drain", "mah", "volt"
    ],
    "Display": [
        "screen", "display", "monitor", "resolution", "brightness",
        "pixel", "lcd", "oled", "touch screen", "panel"
    ],
    "Price": [
        "price", "cost", "expensive", "cheap", "affordable", "value",
        "worth", "money", "budget", "overpriced", "discount"
    ],
    "Packaging": [
        "package", "packaging", "box", "wrap", "wrapped", "unbox",
        "unboxing", "container", "sealed", "damaged box"
    ],
    "Delivery": [
        "delivery", "shipping", "shipped", "arrive", "arrived",
        "transit", "courier", "dispatch", "late", "fast delivery",
        "slow delivery", "tracking"
    ],
    "Customer Service": [
        "seller", "service", "support", "customer", "refund",
        "return", "response", "reply", "contact", "help", "assistance",
        "agent", "helpdesk"
    ],
    "Quality": [
        "quality", "durable", "material", "build", "sturdy",
        "solid", "well made", "poorly made", "craftsmanship",
        "construction", "feels", "texture", "finish"
    ],
    "Performance": [
        "performance", "speed", "fast", "slow", "lag", "smooth",
        "powerful", "efficient", "response", "responsive", "works"
    ],
    "Design": [
        "design", "look", "looks", "aesthetic", "style", "color",
        "colour", "appearance", "beautiful", "ugly", "compact", "slim"
    ],
    "Size": [
        "size", "small", "large", "big", "tiny", "fit", "fits",
        "dimensions", "weight", "heavy", "light", "portable"
    ],
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "aspect_dictionary.json")


def get_aspect_dictionary() -> dict:
    """Return the aspect dictionary."""
    return ASPECT_DICTIONARY


def save_aspect_dictionary(output_path: str = OUTPUT_PATH) -> None:
    """Save the aspect dictionary to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ASPECT_DICTIONARY, f, indent=2)
    print(f"[aspect_dictionary] Saved to {output_path}")


if __name__ == "__main__":
    save_aspect_dictionary()
