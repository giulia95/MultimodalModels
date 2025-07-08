#from transformers import AutoProcessor, mBLIP
from transformers import BlipModel, BlipProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import roc_curve
from numpy import argmax
import yaml
from transformers import AutoProcessor, BlipForConditionalGeneration, CLIPProcessor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

text_model_name = config["model"]["text_model_name"]


class mCLIPClassifier(nn.Module):
    def __init__(self, clip_image_model, clip_text_model):
        super(mCLIPClassifier, self).__init__()
        self.clip_image_model = clip_image_model  # CLIPModel from transformers
        self.clip_text_model = clip_text_model    # SentenceTransformer model

        text_dim = 512
        image_dim = 768

        self.fc = nn.Linear(text_dim + image_dim, 1)

    def forward(self, inputs):
        # -------- TEXT EMBEDDINGS --------

        # SentenceTransformer.forward returns embeddings directly
        text_embeds = self.clip_text_model.forward({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })['sentence_embedding']  # shape: [batch_size, 512]

        # -------- IMAGE EMBEDDINGS --------
        image_outputs = self.clip_image_model.vision_model(
            pixel_values=inputs["pixel_values"]
        ) 

        image_embeds = image_outputs.pooler_output  # shape: [batch_size, 512]

        # print(f"Text hidden dim: {text_embeds.shape[1]}")
        # print(f"Image hidden dim: {image_embeds.shape[1]}")
        
        # -------- COMBINE AND CLASSIFY --------
        combined = torch.cat((text_embeds, image_embeds), dim=1)  # [batch_size, 1024]
        x = self.fc(combined)  # [batch_size, 1]
        return torch.sigmoid(x)

class mBLIPClassifier(nn.Module):
    """
    this model is primarily designed for generation tasks (like captioning or VQA), 
    so it doesn’t natively output standalone "image embedding".
    you can still extract useful embeddings from the intermediate encoder/decoder layers, 
    by passing both the image and text and grabbing hidden states. 
    """
    def __init__(self, blip_model):
        super(mBLIPClassifier, self).__init__()
        self.blip_model = blip_model

        # take text and image embeddings dimension from the model
        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(1408, 512)

        # set the linear layer dimension
        self.fc = nn.Linear(512 + 512, 1)  # ora sarà 1024

    def forward(self, inputs):
        device = next(self.parameters()).device
        #print(device)
        #print(next(self.blip_model.parameters()).device)

        vision_outputs = self.blip_model.vision_model(pixel_values=inputs["pixel_values"])
        image_hidden = vision_outputs.last_hidden_state  # shape: [1, num_patches, hidden_dim]

        # Pool image embeddings (e.g. mean pooling)
        image_embedding = image_hidden.mean(dim=1)  # shape: [1, hidden_dim]

        # Pass full input to model to get decoder hidden states
        outputs = self.blip_model.text_decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=True 
        )        
        # Get text embeddings from decoder hidden states (last layer)
        decoder_hidden = outputs.hidden_states[-1]  # shape: [1, seq_len, hidden_dim]
        text_embedding = decoder_hidden.mean(dim=1)          # shape: [1, hidden_dim]

        # print(f"Text hidden dim: {text_embedding.shape[1]}")
        # print(f"Image hidden dim: {image_embedding.shape[1]}")

        # Optionally normalize the embeddings
        image_embeds = F.normalize(image_embedding, p=2, dim=-1)
        text_embeds = F.normalize(text_embedding, p=2, dim=-1)

        image_proj = self.image_proj(image_embeds) 
        text_proj = self.text_proj(text_embeds) 

        # print(f"Text hidden dim: {text_proj.shape[1]}")
        # print(f"Image hidden dim: {image_proj.shape[1]}")

        combined_embeds = torch.cat((text_proj, image_proj), dim=1).to(device)

        x = self.fc(combined_embeds)
        return torch.sigmoid(x)


def train(model, dataloader, optimizer, criterion, device):
    print("Check GPU in TRAINING")
    print("Model device:", next(model.parameters()).device)
    #print("BLIP device:", next(model.blip_model.parameters()).device)

    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):        
        # Ensure all inputs are on the correct device
        #print(inputs)
        #inputs = {k: v.to(device) for k, v in inputs.items() }
        inputs = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        #print(inputs)
        targets = targets.to(device).float()

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def get_Youden_threshold(targets, pred):
    fpr, tpr, thresholds = roc_curve(targets, pred)
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh


# Collate_fn riceve il batch di elementi e fa:
# - padding dei tensori di testo usando pad_sequence in modo che tutti i tensori del batch abbiano stessa lunghezza
# - crea un unico tensor di immagini per il batch
# - crea un dizionario finale che include i tensori di input_ids, attention_mask e 
#   pixel_values che corrispondono ai dati di testo e immagine per l'intero batch
def collate_fn(batch):

    input_ids = [item[0]['input_ids'].squeeze(0) for item in batch]
    
    attention_mask = [item[0]['attention_mask'].squeeze(0).squeeze(0) for item in batch]  # Removing extra dimension
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    pixel_values = [item[0]['pixel_values'].squeeze(0) for item in batch]
    
    
    pixel_values_padded = torch.stack(pixel_values)

    inputs = {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded, 
            'pixel_values': pixel_values_padded
            }

    labels = torch.tensor([item[1] for item in batch])

    return inputs, labels