from sklearn.model_selection import KFold
from transformers import AutoModel, AutoProcessor, BlipForConditionalGeneration
import gc
import glob
import os
import re
import shutil
import yaml
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Utils import results_organizer
from Utils.classifiers import *
from Utils.data_preprocessing import *
from Utils.checkpoints import (
    sanitize_name,
    load_latest_checkpoint,
    save_checkpoint,
    cleanup_model_checkpoints,
    is_fold_completed,
)
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def count_trainable(module):
    #return number of trainable parameters
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# checkpoint helpers moved to Utils/checkpoints.py


def fold_predictions_path(output_folder, model_name, fold, prefix="preds_"):
    tag = sanitize_name(model_name)
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, f"{prefix}{tag}_fold_{fold}.npz")


def save_fold_predictions(output_folder, model_name, fold, preds, true_labels, indexes):
    path = fold_predictions_path(output_folder, model_name, fold)
    np.savez_compressed(path, preds=np.array(preds), true=np.array(true_labels), idx=np.array(indexes))
    return path


def load_fold_predictions(output_folder, model_name, fold):
    path = fold_predictions_path(output_folder, model_name, fold)
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        return d['preds'].tolist(), d['true'].tolist(), d['idx'].tolist()

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

parser = argparse.ArgumentParser(description="Run multilingual model training using a YAML config file.")
parser.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config file")
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = yaml.safe_load(config_file)

# data
image_folder = config["data"]["image_folder"]
dataset_name = config["data"].get("dataset_name")
data_path = config["data"]["data_path"]
label_path = config["data"].get("label_path", data_path)
label_column = config["data"]["label_column"]
#results
output_folder = config["output"]["main_output_folder"]
# finetuning classifier
epochs = config["model"]["epochs"]
model_processor = config["model"]["processor"]
batch_size = config["model"]["batch_size"]
threshold = config["model"]["threshold"]
save_models = config["model"]["save_models"]
saving_folder = config["model"]["saving_folder"]
text_model_name = config["model"]["text_model_name"]
image_model_name = config["model"]["image_model_name"]
finetune = config["model"].get("finetune", True)
loss_name = str(config["model"].get("loss", "bce")).lower()
focal_gamma = float(config["model"].get("focal_gamma", 2.0))

# Support both nested checkpoint config (model.checkpoin) and legacy flat keys (model.*).
checkpoint_cfg = config["model"].get("checkpoint", {})
checkpoint_enabled = bool(checkpoint_cfg.get("checkpoint_enabled", config["model"].get("checkpoint_enabled", True)))
checkpoint_every_n_epochs = int(checkpoint_cfg.get("checkpoint_every_n_epochs", config["model"].get("checkpoint_every_n_epochs", 1)))
resume_from_checkpoint = bool(checkpoint_cfg.get("resume_from_checkpoint", config["model"].get("resume_from_checkpoint", True)))
checkpoint_dir = checkpoint_cfg.get("checkpoint_dir", config["model"].get("checkpoint_dir", os.path.join(saving_folder, "checkpoints")))
cleanup_checkpoints_on_finish = bool(
    checkpoint_cfg.get("cleanup_checkpoints_on_finish", config["model"].get("cleanup_checkpoints_on_finish", True))
    )


prefix = "fine_tuned_"
print("Results for this model will be saved with the prefix " + prefix + text_model_name.split('/')[1])

print("\tLoading data ...")
data = get_data(data_path, label_path, label_column, dataset_name=dataset_name)
print(f"Finetuning mode: {'full model' if finetune else 'last layers only'}")
if checkpoint_enabled:
    print(f"Checkpointing enabled: every {checkpoint_every_n_epochs} epoch(s)")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Resume from latest checkpoint: {resume_from_checkpoint}")
    print(f"Cleanup checkpoints on successful finish: {cleanup_checkpoints_on_finish}")

if loss_name in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
    criterion = nn.BCEWithLogitsLoss()
    print("Loss function: BCEWithLogitsLoss")
elif loss_name in {"focal", "focal_loss", "focalloss"}:
    criterion = None
    print(f"Loss function: FocalLoss (gamma={focal_gamma}, alpha computed from each training fold)")
else:
    raise ValueError(f"Unsupported loss '{loss_name}'. Use 'bce' or 'focal'.")

print(data.head())

all_predictions = []
all_true_labels = []
indexes = []


print("starting fold-iteration....")
#read folds
kf = KFold(n_splits=10, shuffle=True, random_state=42) #posso fare lo shuffle perchè lo fa sugli indici
fold = 1


for train_index, test_index in kf.split(data):
    gc.collect()

    fold_already_completed = (
        checkpoint_enabled
        and resume_from_checkpoint
        and is_fold_completed(checkpoint_dir, text_model_name, fold, epochs, device)
    )
    if fold_already_completed:
        print(f"Fold {fold} already completed. Reusing the saved checkpoint for inference.")

    # If per-fold predictions were already saved, load them and skip heavy model work
    loaded = load_fold_predictions(output_folder, text_model_name, fold)
    if loaded is not None:
        p, t, idxs = loaded
        all_predictions.extend(p)
        all_true_labels.extend(t)
        # idxs may be numpy types; ensure Python ints
        indexes.extend([int(i) for i in idxs])
        print(f"Loaded saved predictions for fold {fold}, skipping model initialization.")
        fold = fold + 1
        continue

    if text_model_name == 'sentence-transformers/clip-ViT-B-32-multilingual-v1':
        """
        text_model = SentenceTransformer(text_model_name).to(device)
        img_model =  CLIPModel.from_pretrained(image_model_name).to(device)
        classifier = mCLIPClassifier(img_model,text_model).to(device)
        """
        img_model = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        text_model= AutoModel.from_pretrained(text_model_name).to(device)
        processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
        classifier = mCLIPClassifier(img_model, text_model, finetune=finetune).to(device)

        # CALCOLO PARAMETRI
        img_params = count_trainable(img_model)
        text_params = count_trainable(text_model)
        head_params = count_trainable(classifier.fc)
        total_params = img_params + text_params + head_params
        print(f"Trainable params (image): {img_params}")
        print(f"Trainable params (text):  {text_params}")
        print(f"Trainable params (head):  {head_params}")
        print(f"Total trainable params:   {total_params}")
        #sys.exit("Stopping after parameter count as requested.")


        
    elif text_model_name == "Gregor/mblip-mt0-xl":
        #model = BlipModel.from_pretrained(text_model_name).to(device)
        #processor = BlipProcessor.from_pretrained(text_model_name)
        processor = AutoProcessor.from_pretrained(text_model_name)
        model = BlipForConditionalGeneration.from_pretrained(text_model_name).to(device)
        classifier = mBLIPClassifier(model, finetune=finetune).to(device)

        # CALCOLO PARAMETRI
        blip_params = count_trainable(model)
        proj_params = count_trainable(classifier.text_proj) + count_trainable(classifier.image_proj)
        head_params = count_trainable(classifier.fc)
        total_params = blip_params + proj_params + head_params
        print(f"Trainable params (BLIP):  {blip_params}")
        print(f"Trainable params (proj):  {proj_params}")
        print(f"Trainable params (head):  {head_params}")
        print(f"Total trainable params:   {total_params}")
        #sys.exit("Stopping after parameter count as requested.")

        print(classifier.blip_model.device)
    elif text_model_name == "google/siglip-base-patch16-256-multilingual":
        processor = AutoProcessor.from_pretrained(text_model_name)
        model = AutoModel.from_pretrained(text_model_name).to(device)
        classifier = SigLIPClassifier(model, finetune=finetune).to(device)

        # CALCOLO PARAMETRI
        if classifier.fc is None:
            embed_dim = getattr(model.config, "projection_dim", None)
            if embed_dim is None:
                embed_dim = model.config.vision_config.hidden_size
            classifier.fc = nn.Linear(embed_dim * 2, 1).to(device)

        backbone_params = count_trainable(model)
        head_params = count_trainable(classifier.fc)
        total_params = backbone_params + head_params
        print(f"Trainable params (SigLIP backbone): {backbone_params}")
        print(f"Trainable params (head):            {head_params}")
        print(f"Total trainable params:             {total_params}")
        #sys.exit("Stopping after parameter count as requested.")
        

    else: 
            raise ValueError("Unsupported model_type")

    if model_processor:
        print("Dataset Definition with model Processor")
        train_dataset = MemeDataset_processor(data.iloc[train_index], processor, image_folder)
    else:
        print("Dataset Definition without model Processor")
        #train_dataset = MemeDataset(data.iloc[train_index], image_folder)
        train_dataset = MemeDataset_mCLIP(data.iloc[train_index], multilingual_processor, image_folder)

    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print('initializing model ... ')

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4) #optimizes all the parameters of the classifier, including those of both the CLIP model and the fully connected (fc) layer.
    if loss_name in {"focal", "focal_loss", "focalloss"}:
        alpha_fold = compute_alpha(data.iloc[train_index][label_column].values)
        criterion = FocalLoss(alpha=alpha_fold, gamma=focal_gamma)
        print(f"Fold {str(fold)} focal alpha: {alpha_fold:.6f}")

    start_epoch = 0
    if checkpoint_enabled and resume_from_checkpoint:
        latest_checkpoint = load_latest_checkpoint(checkpoint_dir, text_model_name, fold, device)
        if latest_checkpoint is not None:
            classifier.load_state_dict(latest_checkpoint["model_state_dict"])
            optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
            start_epoch = int(latest_checkpoint.get("next_epoch", 0))
            print(f"Resumed fold {fold} from checkpoint {latest_checkpoint['path']} (next epoch: {start_epoch + 1})")

    if fold_already_completed:
        print("Skipping training because this fold has already completed all epochs.")
    else:
        print("training model...")
    for epoch in range(start_epoch, epochs):
        try:
            train_loss = train(classifier, train_loader, optimizer, criterion, device)
        except KeyboardInterrupt:
            if checkpoint_enabled:
                interrupt_ckpt = save_checkpoint(
                    checkpoint_dir,
                    text_model_name,
                    fold,
                    epoch + 1,
                    epoch,
                    classifier,
                    optimizer,
                    interrupted=True,
                )
                print(f"Interrupted during fold {fold}, epoch {epoch + 1}. Saved checkpoint: {interrupt_ckpt}")
            raise

        print(f"Fold {str(fold)}, Epoch {epoch+1}, Training Loss: {train_loss}")
        print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Cached:   ", torch.cuda.memory_reserved() / 1024**2, "MB")

        if checkpoint_enabled and (((epoch + 1) % checkpoint_every_n_epochs == 0) or (epoch + 1 == epochs)):
            ckpt_path = save_checkpoint(
                checkpoint_dir,
                text_model_name,
                fold,
                epoch + 1,
                epoch + 1,
                classifier,
                optimizer,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    if model_processor:
        print("Dataset Definition with model Processor")
        test_set = DataLoader(MemeDataset_processor(data.iloc[test_index], processor, image_folder), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        test_set = DataLoader(MemeDataset_mCLIP(data.iloc[test_index], multilingual_processor, image_folder), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    if threshold == 'Youden':
        print("Estimating JYouden threshold... ")
        # isolate the validation to estimate the best threshold
        validation = Subset(train_dataset, range(int(len(train_dataset) * 0.9), len(train_dataset)))
        validation = DataLoader(
                validation,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(validation):
                inputs = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                targets = targets.to(device).float()
                outputs = classifier(inputs).squeeze(1)
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate everything into a single array
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Compute Youden threshold once on all data
        best_threshold = get_Youden_threshold(all_targets, all_preds)
        print("Estimated Youden Threshold: ", best_threshold)

    # Per-fold collectors so we can save predictions for this fold separately
    fold_preds = []
    fold_trues = []

    with torch.no_grad():
        #probs_0 = []
        #probs_1 = []
        for inputs, targets in tqdm(test_set):
            #print(targets)

            #inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets = targets.to(device).float()
            outputs = classifier(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
        
            if threshold == 'Youden':
                #print("Using Threshold: " + str(best_threshold))
                #threshold = get_Youden_threshold(targets.cpu().numpy(), outputs.cpu().numpy())
                preds = (probs > best_threshold).int()
            else:
                preds = (probs > float(threshold)).int()
            
            cur_preds = preds.cpu().numpy().tolist()
            cur_trues = targets.cpu().numpy().tolist()
            all_predictions.extend(cur_preds)
            all_true_labels.extend(cur_trues)
            fold_preds.extend(cur_preds)
            fold_trues.extend(cur_trues)

            """
            for item in outputs:
                probs_0.append(1- item.detach().cpu().item())
                probs_1.append(item.detach().cpu().item())
            """

    indexes.extend(test_index)

    # Save per-fold predictions so future runs can load them instead of re-running inference
    try:
        saved_path = save_fold_predictions(output_folder, text_model_name, fold, fold_preds, fold_trues, test_index)
        print(f"Saved fold predictions: {saved_path}")
    except Exception as e:
        print(f"Warning: failed to save fold predictions for fold {fold}: {e}")

    if save_models:
            # Optionally save the models
            a = str(text_model_name.split('/')[1])
            classifier.save(f'{saving_folder}_{prefix}_{a}_classifier_{str(fold)}.h5')

    fold = fold +1


macro_f1 = results_organizer.save_performances_on_file(output_folder, prefix+"_", text_model_name, all_true_labels, all_predictions)

print(data.shape)
print(len(all_predictions))
print(len(all_true_labels))

results_organizer.save_predictions_on_file(output_folder, data_path, data, prefix+"_" + text_model_name, all_predictions, all_true_labels, label_column, indexes)

if checkpoint_enabled and cleanup_checkpoints_on_finish:
    removed_dir = cleanup_model_checkpoints(checkpoint_dir, text_model_name)
    if removed_dir:
        print(f"Checkpoint cleanup complete. Removed: {removed_dir}")
    else:
        print("Checkpoint cleanup skipped: no checkpoint directory found for this model.")