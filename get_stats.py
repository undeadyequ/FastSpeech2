import json


iiv_embs = {
    "Neutral": {
        "0": "0019_000001.npy",
        "1": ""
    },
    "Sad": {
        "0": "",
        "1": "",
        "2": "",
        "3": ""
    },
    "Angry": {
        "0": "0019_000368.npy",
        "1": "",
        "2": ""
    },
    "Surprise": {
        "0": "",
        "1": ""
    },
    "Happy": {
        "0": "",
        "1": "0019_000718.npy"
    }
}

idx_emo_dict_f = "/home/rosen/project/FastSpeech2/ESD/metadata_new.json"

with open(idx_emo_dict_f, "r") as f:
    meta_data_dict = json.load(f)

# get iiv_embs

for id, attr_dict in meta_data_dict.items():
    emo = attr_dict["emotion"]
    grp = attr_dict["group"]
    iiv_embs[emo][grp] = id + ".npy"

print(json.dumps(iiv_embs, indent=4))