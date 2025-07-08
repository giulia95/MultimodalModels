from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from sklearn.utils import shuffle 
from torchvision import transforms
from transformers import AutoProcessor, BlipForConditionalGeneration

def import_Moxy(data_path, cols = ['image_name', 'text', 'GS _ Esperto']):
  GS_Ita = pd.read_excel(data_path, sheet_name='Ita', usecols=cols)
  GS_Esp = pd.read_excel(data_path, sheet_name='Esp', usecols=cols) 

  GS_Esp['label'] = GS_Esp.apply(lambda x: 1 if x['GS _ Esperto']== "Sì" else 0, axis=1)
  GS_Ita['label'] = GS_Ita.apply(lambda x: 1 if x['GS _ Esperto']== "Sì" else 0, axis=1)

  # Concatenate the DataFrames
  MOxy = shuffle(pd.concat([GS_Ita, GS_Esp], ignore_index=True), random_state=42).reset_index(drop=True)
  return MOxy

def import_MAMIta(data_path, cols = ['Meme', 'image_name', 'Agreement', 'text']):
  data = pd.read_csv(data_path, sep='\t', usecols=cols)
  data['label'] = data['Agreement']
  data=data.drop(columns=['Agreement'])
  return data


class MemeDataset(Dataset):
    def __init__(self, dataframe, image_folder):
        self.dataframe = dataframe
        self.image_folder = image_folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image_name'])
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format

        #print(image.shape)
        text = str(row['text']) if pd.notna(row['text']) else ""
        label = row['label']
        
        inputs = {"text":text, "image": image}  

        labels = torch.tensor(label.astype(int), dtype=torch.float32)
        return inputs, labels

class MemeDataset_processor(Dataset):
    def __init__(self, dataframe, processor,image_folder):
        self.dataframe = dataframe
        self.processor = processor
        self.image_folder = image_folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image_name']) # Assicurati che l'immagine sia RGB
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        
        #print(image.shape)
        text = str(row['text']) if pd.notna(row['text']) else ""

        # Preprocess both image and text
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)

        disagreement = torch.tensor(row['label'].astype(int), dtype=torch.float32)
        return inputs, disagreement
