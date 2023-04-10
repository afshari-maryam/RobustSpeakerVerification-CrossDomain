<<<<<<< HEAD
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
print("January....February...March....April....")
print("Date: 10-April-2023")
print("Evaluate Speaker Verification of the 2 audio from dataset with the WavLM model")

import os
import gradio as gr
import torch
import pydub
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
import itertools

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

#------------------this Part add for voxceleb reading------------------
#import os
# Using readlines()
file1 = open('veri_test.txt', 'rb')
Lines = file1.readlines()

labels =[]
speaker1 =[]
speaker2 =[]

for line in Lines:
  temp_list1 = line.split()
  labels.append(int(temp_list1[0]))
  speaker1.append(str(temp_list1[1].decode("utf-8")))
  speaker2.append(str(temp_list1[2].decode("utf-8")))


#for line in Lines:
#  temp_list1 = line.split()
#  labels.append(int(temp_list1[0]))
#  speaker1.append(str(temp_list1[1]))
#  speaker2.append(str(temp_list1[2]))

#check whether ok or not?
#print(labels[0])
print(f"labels[0] =  {labels[0]}")
print(f"speaker1[0] =  {speaker1[0]}")
print(f"speaker2[0] =  {speaker2[0]}")


root = "/mnt/disk1/data/voxceleb_1.1/wav/"
#get_audio_1-----------------------------------------
#list1 =Lines[0].split()
file_path1 = os.path.join(root, speaker1[0])
#print(file_path1)
print(f"file audio 1 =  {file_path1}")

#get_audio_2
#list2 =Lines[1].split()
#/mnt/disk1/data/voxceleb_1.1
#code_file_Path : /home/afshari/project2/PretrainWav2vec2

#file2 = list2[2]
#file_path2 = os.path.join(root, str(file2))
file_path2 = os.path.join(root, str(speaker2[0]))
#print(file_path2)
print(f"file audio 2 =  {file_path2}")

print("similarity function 2:")
similarity_fn(file_path1,file_path2)

#Third phase---------------------------------------------

print("Second phase :------------------------------------------------------------------------")
# Set the model to evaluation mode
#model.eval()

# Initialize an empty score list to store the similarities
scores = []
root = "/mnt/disk1/data/voxceleb_1.1/wav/"

for (path1,path2) in zip(speaker1,speaker2):
  
  file_path1 = os.path.join(root, path1)
  file_path2 = os.path.join(root, path2)
  
  wav1, sr1 = load_audio(file_path1)
  print("wav1 details : ")
  print(wav1, wav1.shape, wav1.dtype)
  wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS)
  wav2, sr2 = load_audio(file_path2)
  wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS)


  print("wav1 and 2 shape : ")
  print(wav1.shape, wav2.shape)

  ## Preprocess the waveforms
  #waveform1 = waveform1.squeeze(0).numpy()
  #waveform2 = waveform2.squeeze(0).numpy()
  #waveform1, _ = apply_effects_tensor(torch.tensor(waveform1).unsqueeze(0), waveform1.sample_rate, EFFECTS)
  #waveform2, _ = apply_effects_tensor(torch.tensor(waveform2).unsqueeze(0), waveform2.sample_rate, EFFECTS)


  # Extract features from the waveforms
  input1 = feature_extractor(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)
  input2 = feature_extractor(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)

  # Compute the embeddings for the waveforms
  with torch.no_grad():
      emb1 = model(input1).embeddings
      emb2 = model(input2).embeddings
  emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
  emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()

  # Compute the similarity between the embeddings
  similarity = cosine_sim(emb1, emb2).numpy()[0]

  # Add the similarity to the list score
  scores.append(similarity)

print(f"scores[0] = {scores[0]}")
=======
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
print("January....February...March....April....")
print("Date: 10-April-2023")
print("Evaluate Speaker Verification of the 2 audio from dataset with the WavLM model")

import os
import gradio as gr
import torch
import pydub
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
import itertools

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

#------------------this Part add for voxceleb reading------------------
#import os
# Using readlines()
file1 = open('veri_test.txt', 'rb')
Lines = file1.readlines()

labels =[]
speaker1 =[]
speaker2 =[]

for line in Lines:
  temp_list1 = line.split()
  labels.append(int(temp_list1[0]))
  speaker1.append(str(temp_list1[1].decode("utf-8")))
  speaker2.append(str(temp_list1[2].decode("utf-8")))


#for line in Lines:
#  temp_list1 = line.split()
#  labels.append(int(temp_list1[0]))
#  speaker1.append(str(temp_list1[1]))
#  speaker2.append(str(temp_list1[2]))

#check whether ok or not?
#print(labels[0])
print(f"labels[0] =  {labels[0]}")
print(f"speaker1[0] =  {speaker1[0]}")
print(f"speaker2[0] =  {speaker2[0]}")


root = "/mnt/disk1/data/voxceleb_1.1/wav/"
#get_audio_1-----------------------------------------
#list1 =Lines[0].split()
file_path1 = os.path.join(root, speaker1[0])
#print(file_path1)
print(f"file audio 1 =  {file_path1}")

#get_audio_2
#list2 =Lines[1].split()
#/mnt/disk1/data/voxceleb_1.1
#code_file_Path : /home/afshari/project2/PretrainWav2vec2

#file2 = list2[2]
#file_path2 = os.path.join(root, str(file2))
file_path2 = os.path.join(root, str(speaker2[0]))
#print(file_path2)
print(f"file audio 2 =  {file_path2}")

print("similarity function 2:")
similarity_fn(file_path1,file_path2)

#Third phase---------------------------------------------

print("Second phase :------------------------------------------------------------------------")
# Set the model to evaluation mode
#model.eval()

# Initialize an empty score list to store the similarities
scores = []
root = "/mnt/disk1/data/voxceleb_1.1/wav/"

for (path1,path2) in zip(speaker1,speaker2):
  
  file_path1 = os.path.join(root, path1)
  file_path2 = os.path.join(root, path2)
  
  wav1, sr1 = load_audio(file_path1)
  print("wav1 details : ")
  print(wav1, wav1.shape, wav1.dtype)
  wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS)
  wav2, sr2 = load_audio(file_path2)
  wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS)


  print("wav1 and 2 shape : ")
  print(wav1.shape, wav2.shape)

  ## Preprocess the waveforms
  #waveform1 = waveform1.squeeze(0).numpy()
  #waveform2 = waveform2.squeeze(0).numpy()
  #waveform1, _ = apply_effects_tensor(torch.tensor(waveform1).unsqueeze(0), waveform1.sample_rate, EFFECTS)
  #waveform2, _ = apply_effects_tensor(torch.tensor(waveform2).unsqueeze(0), waveform2.sample_rate, EFFECTS)


  # Extract features from the waveforms
  input1 = feature_extractor(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)
  input2 = feature_extractor(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)

  # Compute the embeddings for the waveforms
  with torch.no_grad():
      emb1 = model(input1).embeddings
      emb2 = model(input2).embeddings
  emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
  emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()

  # Compute the similarity between the embeddings
  similarity = cosine_sim(emb1, emb2).numpy()[0]

  # Add the similarity to the list score
  scores.append(similarity)

print(f"scores[0] = {scores[0]}")
>>>>>>> 9892cc1755e67ca31189e18bd99448c6148e0754
