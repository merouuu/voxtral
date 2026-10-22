import csv, random
from pathlib import Path

EMOTIONS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
            "HAP": "happiness", "NEU": "neutral", "SAD": "sadness"}
SENTENCES = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.", "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.", "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}

def load_clips(root):
    root = Path(root)
    with open(root / "VideoDemographics.csv") as f:
        gender = {int(r["ActorID"]): r["Sex"].lower() for r in csv.DictReader(f)}
    clips = []
    for p in sorted((root / "AudioWAV").glob("*.wav")):
        actor, sentence, emo, _ = p.stem.split("_")
        clips.append({"path": p, "actor": int(actor), "sentence": sentence,
                      "emotion": EMOTIONS[emo], "gender": gender[int(actor)]})
    return clips

def split_by_actor(clips, test_frac=0.3, seed=0):
    actors = sorted({c["actor"] for c in clips})
    random.Random(seed).shuffle(actors)
    test_actors = set(actors[: round(len(actors) * test_frac)])
    train = [c for c in clips if c["actor"] not in test_actors]
    test = [c for c in clips if c["actor"] in test_actors]
    return train, test
