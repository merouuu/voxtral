import csv, random
from pathlib import Path

EMOTIONS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
            "HAP": "happiness", "NEU": "neutral", "SAD": "sadness"}

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
