# Import the transformers library
import transformers
import datasets
import os
import gradio as gr
import torch
import pydub
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
import itertools
import numpy as np  
import pandas as pd  
  
     
#from sklearn.metrics import roc_curve, eer

# Print the version of the transformers library
print(transformers.__version__)
print(datasets.__version__)
print(gr.__version__)
#print(pydub.__version__)
print("In the name of GOD")
print("Date1: 2-Janyuary-2023")
print("Date2: 9-Janyuary-2023")
print("Date3: 10-April-2023")
print("Evaluate the whole of dataset on the WavLM model")

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
    #print(wav1, wav1.shape, wav1.dtype)
    wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS)
    wav2, sr2 = load_audio(path2)
    wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS)
    #print(wav1.shape, wav2.shape)

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


def compute_scores(spkr_list1,spkr_list2):
  # Initialize an empty score list to store the similarities
  scores = []
  root = "/mnt/disk1/data/voxceleb_1.1/wav/"
  count = 0
  print("comput scores function starts ....................................")
  print(".............................. ....................................")
  print(".............................. ....................................")
  for (path1,path2) in zip(spkr_list1,spkr_list2):
    count = count+1
    print(f"number = {count}")
    file_path1 = os.path.join(root, path1)
    file_path2 = os.path.join(root, path2)
    # Add the similarity to the list score
    scores.append(similarity_fn(file_path1,file_path2))
  print("comput scores function ended ....................................")
  return scores


def write_score_in_csv_file(input_score):
    # list score 
    scr = input_score
    # dictionary of lists  
    dict = {'score': scr}  
       
    df = pd.DataFrame(dict) 
    
    # saving the dataframe 
    df.to_csv('my_score_cfg.csv') 

#------------------this Part add for voxceleb reading------------------
#import os
# Using readlines()
file1 = open('veri_test.txt', 'rb')
Lines = file1.readlines()

root = "/mnt/disk1/data/voxceleb_1.1/wav/"
labels =[]
speaker1 =[]
speaker2 =[]

for line in Lines:
  temp_list1 = line.split()
  labels.append(int(temp_list1[0]))
  speaker1.append(str(temp_list1[1].decode("utf-8")))
  speaker2.append(str(temp_list1[2].decode("utf-8")))    

print(f"labels[0] =  {labels[0]}")
print(f"speaker1[0] =  {speaker1[0]}")
print(f"speaker2[0] =  {speaker2[0]}")

#-----------------call and check similarity_fn--------------------------------------------------

file_path1 = os.path.join(root, speaker1[0])
print(f"file audio 1 =  {file_path1}")
file_path2 = os.path.join(root, str(speaker2[0]))
print(f"file audio 2 =  {file_path2}")
print("similarity function :")
similarity_fn(file_path1,file_path2)

#------------------call compute score functions ------------------------------------------------

my_scores =compute_scores(speaker1,speaker2)
print(f"my_scores[0] = {my_scores[0]}")

#save score in a csv file
print("write scores in a csv file using pandas")
write_score_in_csv_file(my_scores)


## Convert the labels to a numpy array
#labels = np.array([label1 == label2 for _, label1 in dataset for _, label2 in dataset])
labels = np.array(labels)

# Calculate the false positive rate and true positive rate
fpr, tpr, _ = roc_curve(labels, my_scores)

# Calculate the EER
eer = eer(fpr, tpr)

print(f'Equal Error Rate (EER): {eer:.4f}')