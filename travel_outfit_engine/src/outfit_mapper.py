OUTFIT_MAP = {
    "Casual": {
        "hot":  ["Linen Shirt", "Chino Shorts", "Canvas Loafers"],           # 75 °F +
        "warm": ["Linen Shirt", "Slim Chinos", "Canvas Sneakers"],            # 60–74 °F
        "cool": ["Light Knit Sweater", "Straight Jeans", "Clean Sneakers"],   # 45–59 °F
        "cold": ["Wool Sweater", "Slim Trousers", "Leather Boots", "Winter Coat"],  # < 45 °F
    },
    "Rain Ready": {
        "hot":  ["Light Rain Jacket", "Quick-dry Shorts", "Water-resistant Shoes"],
        "warm": ["Waterproof Jacket", "Slim Chinos", "Water-resistant Shoes"],
        "cool": ["Waterproof Jacket", "Waterproof Trousers", "Waterproof Boots"],
        "cold": ["Heavy Waterproof Parka", "Waterproof Trousers", "Insulated Waterproof Boots"],
    },
    "Formal": {
        "hot":  ["Light Linen Blazer", "Dress Trousers", "Leather Loafers"],
        "warm": ["Blazer", "Dress Pants", "Leather Oxford Shoes"],
        "cool": ["Wool Blazer", "Dress Pants", "Leather Oxford Shoes", "Overcoat"],
        "cold": ["Heavy Wool Suit", "Dress Pants", "Leather Oxford Shoes", "Wool Overcoat", "Scarf"],
    },
    "Athletic": {
        "hot":  ["Moisture-wicking Tee", "Running Shorts", "Running Shoes"],
        "warm": ["Sports Tee", "Track Pants", "Running Shoes"],
        "cool": ["Long-sleeve Base Layer", "Jogger Pants", "Running Shoes", "Light Jacket"],
        "cold": ["Thermal Base Layer", "Insulated Joggers", "Running Shoes", "Fleece Jacket"],
    },
}


def _temp_band(temp_f: float) -> str:
    if temp_f >= 75:
        return "hot"
    elif temp_f >= 60:
        return "warm"
    elif temp_f >= 45:
        return "cool"
    else:
        return "cold"


def get_items(category: str, temp_f: float) -> list:
    """Return temperature-appropriate clothing items for the given category."""
    mapping = OUTFIT_MAP.get(category, {})
    band    = _temp_band(temp_f)
    # Fall through bands if the exact one is missing
    for b in (band, "warm", "hot"):
        items = mapping.get(b)
        if items:
            return items
    return []
