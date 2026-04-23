from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import os
import torch
from sklearn.utils import shuffle 
from torchvision import transforms
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, CLIPImageProcessor

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

def soft_labels(predictions, annotators_number = 6):
  if 'YES' in predictions:
    return predictions.count('YES')/annotators_number
  else:
    return 0

def most_frequent(List): #hard label
  counter = 0
  value = List[0]

  for i in List:
      curr_frequency = List.count(i)
      if(curr_frequency> counter):
          counter = curr_frequency
          value = i
  # if 50% agreement, than positive label
  if counter == 3:
    value = 'YES'
  return value, counter

def get_dataset_labels(df, dataset_name='MAMI', columns = ['file_name','text','misogynous','soft_label_0','soft_label_1', 'label']):
  """
  df: dataframe to elaborate
  colums: list of output columns
  ______________________________
  Extract two columns from the soft-label column to represent disagreement on the positive and negative label.
  Add a "disagreemen" column with boolean values (1 for agreement, 0 for disagreement)
  Rename the column "text" in "original text" to distiguish with the token-column "text"
  """
  if dataset_name=='MAMI':
    df['soft_label_1']= df['NOTmisogynous'].apply(lambda x: (3-x)/3)
    df['soft_label_0']= df['NOTmisogynous'].apply(lambda x: x/3)
    df.rename({'Text Transcription': 'text'}, axis=1, inplace=True)
  elif dataset_name =="EXIST":
    df['soft_label_1']= df['labels_task4'].apply(lambda x: soft_labels(x))
    df['soft_label_0']= df['soft_label_1'].apply(lambda x: 1-x)
    df['misogynous'] = df['labels_task4'].apply(lambda x: most_frequent(x)[0])
    df['misogynous'] = df['misogynous'].map({'YES': 1, 'NO': 0})
    df['path_memes'] = df['path_memes'].apply(lambda x: x.split('/')[1])
    df.rename(columns={
        'path_memes': 'file_name',
        #'text': 'original_text'
    }, inplace=True)
  df['label'] = df['soft_label_0'].apply(lambda x : int(x==0 or x==1)) #1 = agreement
  return df[columns]

def load_data(data_path):
    # Validate paths
    input_csv_path = Path(data_path)
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
    
    if os.path.splitext(data_path)[1].lower() == ".csv":
        return pd.read_csv(data_path)
    elif os.path.splitext(data_path)[1].lower() in [".xlsx", ".xls"]:
        return pd.read_excel(data_path)
    elif os.path.splitext(data_path)[1].lower() in [".json"]:
        return pd.read_json(data_path, orient='index')
    else:
        raise ValueError(f"Unsupported file extension: {os.path.splitext(data_path)[1].lower()}")
    

def get_data(data_path, labels_path, label_columns):
    data = load_data(data_path)
    len_data = len(data)
    if data_path != labels_path:
        #print(load_data(labels_path)[label_columns].head())
        data = data.merge(load_data(labels_path)[label_columns], left_on='file_name', right_on='meme').drop_duplicates().reset_index()
        if len_data != len(data):
            print(f"Warning: data dimention is changed from {len_data} to {len_data}. Check input data. ")
    
    dataset_name = data_path.split("data/")[1].split("/")[0]
    data = get_dataset_labels(data, dataset_name)

    return data

class MemeDataset(Dataset):
    def __init__(self, dataframe, image_folder):
        self.dataframe = dataframe
        self.image_folder = image_folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row['file_name'])
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format

        #print(image.shape)
        text = str(row['text']) if pd.notna(row['text']) else ""
        label = row['label']
        
        inputs = {"text":text, "image": image}  

        labels = torch.tensor(label.astype(int), dtype=torch.float32)
        return inputs, labels

def multilingual_processor(texts, images):
    """
    Accepts lists of texts and PIL images.
    Returns a dict of input tensors for SentenceTransformer.forward().
    """
    # Load tokenizer from the multilingual CLIP model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/clip-ViT-B-32-multilingual-v1")

    # Load image processor from the original CLIP model
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    encoded_images = image_processor(images=images, return_tensors="pt")
    return {**encoded_texts, **encoded_images}

# for mCLIP
class MemeDataset_mCLIP(Dataset):
    def __init__(self, dataframe, processor, image_folder):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = os.path.join(self.image_folder, row['file_name'])
        image = Image.open(image_path).convert("RGB")

        text = str(row['text']) if pd.notna(row['text']) else ""
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Use processor to preprocess text and image (batch of size 1)
        processed = self.processor([text], [image])

        # Remove batch dimension (B=1)
        processed = {k: v.squeeze(0) for k, v in processed.items()}

        return processed, label

# for BLIP
class MemeDataset_processor(Dataset):
    def __init__(self, dataframe, processor,image_folder):
        self.dataframe = dataframe
        self.processor = processor
        self.image_folder = image_folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row['file_name']) # Assicurati che l'immagine sia RGB
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        
        #print(image.shape)
        text = str(row['text']) if pd.notna(row['text']) else ""

        # Add truncation=True and max_length to handle sequence length limits
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True, max_length=64)

        disagreement = torch.tensor(row['label'].astype(int), dtype=torch.float32)
        return inputs, disagreement
