DANGEROUS_SOUNDS = {
    "explosion": ["Explosion", "Blast", "Bomb"],
    "gun": ["Gunshot", "Gun fire", "Shooting"],
    "scream": ["Scream", "Shout", "Yell"],
    "alarm": ["Alarm", "Siren", "Warning"],
    "fire": ["Fire", "Burning", "Smoke alarm"],
    "crash": ["Crash", "Collision", "Breaking glass"],
    "fight": ["Fight", "Violence", "Struggle"]
}

MODEL_CONFIG = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000
}