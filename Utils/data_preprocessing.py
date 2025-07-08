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


from transformers import AutoTokenizer, CLIPImageProcessor
# Load tokenizer from the multilingual CLIP model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/clip-ViT-B-32-multilingual-v1")

# Load image processor from the original CLIP model
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def multilingual_processor(texts, images):
#     """
#     Accepts lists of texts and PIL images.
#     Returns a dict of input tensors for SentenceTransformer.forward().
#     """
#     encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#     encoded_images = image_processor(images=images, return_tensors="pt")
#     return {**encoded_texts, **encoded_images}

from typing import Union, List
def multilingual_processor(
    text: Union[str, List[str]],
    images: Union[Image.Image, List[Image.Image]],
    return_tensors: str = "pt",
    padding: Union[bool, str] = True,
    truncation: bool = True,
    max_length: int = 77,
):
    """
    Preprocess multilingual text and images like CLIPProcessor does.

    Supports both single and batched inputs.

    Args:
        text (str or List[str]): Text or list of texts.
        images (PIL.Image or List[PIL.Image]): Image or list of images.
        return_tensors (str): Output tensor format ("pt", "np", etc.).
        padding (bool or str): Padding strategy.
        truncation (bool): Truncate text inputs.
        max_length (int): Max text sequence length.

    Returns:
        Dict[str, torch.Tensor]: Combined preprocessed inputs for CLIP model.
    """

    # Normalize to list
    if isinstance(text, str):
        text = [text]
    if isinstance(images, Image.Image):
        images = [images]

    encoded_text = tokenizer(
        text,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors
    )

    encoded_images = image_processor(
        images=images,
        return_tensors=return_tensors
    )

    processed = {**encoded_text, **encoded_images}
    return {k: v.squeeze(0) for k, v in processed.items()} 

# for mCLIP OLD
# class MemeDataset_mCLIP(Dataset):
#     def __init__(self, dataframe, processor, image_folder):
#         self.dataframe = dataframe
#         self.image_folder = image_folder
#         self.processor = processor

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]

#         image_path = os.path.join(self.image_folder, row['image_name'])
#         image = Image.open(image_path).convert("RGB")

#         text = str(row['text']) if pd.notna(row['text']) else ""
#         label = torch.tensor(row['label'], dtype=torch.float32)

#         # Use processor to preprocess text and image (batch of size 1)
#         processed = self.processor([text], [image])

#         # Remove batch dimension (B=1)
#         processed = {k: v.squeeze(0) for k, v in processed.items()}

#         return processed, label


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
        image_path = os.path.join(self.image_folder, row['image_name']) # Assicurati che l'immagine sia RGB
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        
        #print(image.shape)
        text = str(row['text']) if pd.notna(row['text']) else ""

        # Preprocess both image and text
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)

        disagreement = torch.tensor(row['label'].astype(int), dtype=torch.float32)
        return inputs, disagreement
