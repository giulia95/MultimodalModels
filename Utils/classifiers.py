import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import roc_curve
from numpy import argmax
import yaml
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

text_model_name = config["model"]["text_model_name"]


class mCLIPClassifier(nn.Module):
    def __init__(self, clip_image_model, clip_text_model, finetune=False):
        super(mCLIPClassifier, self).__init__()
        self.clip_image_model = clip_image_model  # CLIPModel from transformers
        self.clip_text_model = clip_text_model    # SentenceTransformer model

        text_dim = 768
        image_dim = 768

        self.fc = nn.Linear(text_dim + image_dim, 1)

        # No need to set requires_grad=True because it's True by default
        if not finetune:
            for param in self.clip_image_model.parameters():
                param.requires_grad = False
            for param in self.clip_text_model.parameters():
                param.requires_grad = False

    def forward(self, inputs):

        # -------- TEXT EMBEDDINGS --------
        # SentenceTransformer.forward returns embeddings directly
        text_outputs = self.clip_text_model.forward(
            input_ids=inputs["input_ids"],         # tensor [batch_size, seq_len]
            attention_mask=inputs["attention_mask"]  # tensor [batch_size, seq_len]
            )
        last_hidden = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # CLS token: first token embedding
        text_embeds = last_hidden[:, 0, :]           # [batch_size, hidden_size]

        # -------- IMAGE EMBEDDINGS --------
        image_outputs = self.clip_image_model.vision_model(
            pixel_values=inputs["pixel_values"]
        ) 

        image_embeds = image_outputs.pooler_output  # shape: [batch_size, 512]
        
        # -------- COMBINE AND CLASSIFY --------
        combined = torch.cat([text_embeds, image_embeds], dim=1)  # [batch_size, 1024]
        logits = self.fc(combined)  # [batch_size, 1]
        return logits

class mBLIPClassifier(nn.Module):
    """
    this model is primarily designed for generation tasks (like captioning or VQA), 
    so it doesn’t natively output standalone "image embedding".
    you can still extract useful embeddings from the intermediate encoder/decoder layers, 
    by passing both the image and text and grabbing hidden states. 
    """
    def __init__(self, blip_model, finetune=False):
        super(mBLIPClassifier, self).__init__()
        self.blip_model = blip_model

        # take text and image embeddings dimension from the model
        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(1408, 512)

        # set the linear layer dimension
        self.fc = nn.Linear(512 + 512, 1)  # ora sarà 1024

        # If finetune is False, train only projection and classification layers.
        if not finetune:
            for param in self.blip_model.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        device = next(self.parameters()).device

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

        # Optionally normalize the embeddings
        image_embeds = F.normalize(image_embedding, p=2, dim=-1)
        text_embeds = F.normalize(text_embedding, p=2, dim=-1)

        image_proj = self.image_proj(image_embeds) 
        text_proj = self.text_proj(text_embeds) 

        combined_embeds = torch.cat((text_proj, image_proj), dim=1).to(device)

        logits = self.fc(combined_embeds)
        return logits


class SigLIPClassifier(nn.Module):
    def __init__(self, siglip_model, finetune=False):
        super(SigLIPClassifier, self).__init__()
        self.siglip_model = siglip_model
        self.fc = None  # Will be initialized on first forward pass
        self.device_set = False

        # Freeze model parameters if not finetuning
        if not finetune:
            for param in self.siglip_model.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        outputs = self.siglip_model(**inputs, return_dict=True)

        text_embeds = outputs.text_embeds  
        image_embeds = outputs.image_embeds

        # Initialize fc layer on first forward pass with actual embedding dimensions
        if self.fc is None:
            combined_dim = text_embeds.shape[-1] + image_embeds.shape[-1]
            self.fc = nn.Linear(combined_dim, 1).to(text_embeds.device)
            print(f"Initialized fc layer with {combined_dim} input features")

        # Concatenate correctly
        combined_embeds = torch.cat((text_embeds, image_embeds), dim=-1)

        logits = self.fc(combined_embeds)  # Linear layer
        return logits


def train(model, dataloader, optimizer, criterion, device):
    print("Check GPU in TRAINING")
    print("Model device:", next(model.parameters()).device)
    #print("BLIP device:", next(model.blip_model.parameters()).device)

    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):        
        inputs = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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

def collate_fn(batch):
    
    if text_model_name =='sentence-transformers/clip-ViT-B-32-multilingual-v1':
        inputs = {
            'text': [item[0]['text'] for item in batch],
            'image': [item[0]['image'] for item in batch]
        }

    else:
        # Handle processor outputs which may have different structures
        first_item = batch[0][0]
        
        inputs = {}
        
        # Handle input_ids
        if 'input_ids' in first_item:
            input_ids = [item[0]['input_ids'].squeeze(0) for item in batch]
            inputs['input_ids'] = pad_sequence(input_ids, batch_first=True, padding_value=0)
        
        # Handle attention_mask (optional, not all processors include it)
        if 'attention_mask' in first_item:
            attention_mask = [item[0]['attention_mask'].squeeze(0) if item[0]['attention_mask'].dim() > 1 else item[0]['attention_mask'] for item in batch]
            inputs['attention_mask'] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # Handle pixel_values
        if 'pixel_values' in first_item:
            pixel_values = [item[0]['pixel_values'].squeeze(0) if item[0]['pixel_values'].dim() > 3 else item[0]['pixel_values'] for item in batch]
            inputs['pixel_values'] = torch.stack(pixel_values)

    labels = torch.tensor([item[1] for item in batch])

    return inputs, labels