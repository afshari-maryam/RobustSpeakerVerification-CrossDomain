# Import the transformers library
import transformers
import datasets
import gradio as gr
import pydub

# Print the version of the transformers library
print(transformers.__version__)
print(datasets.__version__)
print(gr.__version__)
#print(pydub.__version__)
print("In the name of GOD")
print("Date: 2-Janyuary-2023")
print("Ya Zahra")
print("well come back Date: 15-February-2023")
print("Ya Hossein")
print("well come back Date:  February .... March.....April.......10-April-2023")
print("Check the transformers library.")
import os
import gradio as gr
import torch
import pydub
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(f"I'm learning {language} from {school}.")
print(f"device = {device} .")

def load_audio(file_name):
    audio = pydub.AudioSegment.from_file(file_name)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate


EFFECTS = [
    ["remix", "-"],
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-1.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    ["trim", "0", "10"],
]

THRESHOLD = 0.85

model_name = "microsoft/wavlm-base-plus-sv"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=-1)


def similarity_fn(path1, path2):
    if not (path1 and path2):
        return '<b style="color:red">ERROR: Please record audio for *both* speakers!</b>'
    
    wav1, sr1 = load_audio(path1)
    print(wav1, wav1.shape, wav1.dtype)
    wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS)
    wav2, sr2 = load_audio(path2)
    wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS)
    print(wav1.shape, wav2.shape)

    input1 = feature_extractor(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)
    input2 = feature_extractor(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        emb1 = model(input1).embeddings
        emb2 = model(input2).embeddings
    emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
    emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()
    similarity = cosine_sim(emb1, emb2).numpy()[0]

    if similarity >= THRESHOLD:
        #output = OUTPUT_OK.format(similarity * 100)
        output = (similarity * 100)
    else:
        #output = OUTPUT_FAIL.format(similarity * 100)
        output = (similarity * 100)

    return output


inputs = [
    gr.inputs.Audio(source="microphone", type="filepath", optional=True, label="Speaker #1"),
    gr.inputs.Audio(source="microphone", type="filepath", optional=True, label="Speaker #2"),
]

print("Hey maryam ^..^ I'm good! Thank God:)")

my_path1 ="/mnt/disk1/data/voxceleb_1.1/wav/id10270/x6uYqmx31kE/00001.wav"
my_path2 ="/mnt/disk1/data/voxceleb_1.1/wav/id10270/8jEAjG6SegY/00008.wav"

similarity_fn(my_path1,my_path2)