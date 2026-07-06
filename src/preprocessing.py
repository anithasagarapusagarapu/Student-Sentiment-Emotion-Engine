import re

def clean_text(text: str) -> str:
    """Standardizes text by removing URLs and extra characters."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(re.compile(r'https?://\S+|www\.\S+'), '', text)
    text = re.sub(re.compile(r"[^a-zA-Z0-9\s?!'.]"), '', text)
    return text

def apply_keyword_enhancement(text: str, model_probabilities: dict) -> dict:
    """Adjusts prediction scores based on specific exact whole-word keywords."""
    cleaned = text.lower()
    adjusted_probs = model_probabilities.copy()
    
    rules = {
        "Confused": ["lost", "stuck", "confused", "clueless"],
        "Frustrated": ["hate", "broken", "useless", "crashing", "annoying", "quit"],
        "Bored": ["dry", "boring", "sleepy", "tired", "uninteresting"],
        "Confident": ["easy", "solved", "perfectly", "simple", "ready"],
        "Curious": ["wonder", "explore", "excited", "how does"]
    }
    
    # Use explicit word boundaries (\b) to avoid partial sub-word matching mixups
    for emotion, keywords in rules.items():
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, cleaned):
                if emotion in adjusted_probs:
                    adjusted_probs[emotion] += 0.25
            
    total_score = sum(adjusted_probs.values())
    if total_score > 0:
        adjusted_probs = {k: round(v / total_score, 4) for k, v in adjusted_probs.items()}
        
    return adjusted_probs