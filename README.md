# MultimodalModels

This project trains a multimodal meme classifier that combines text and image features. The main entry point is [multilingual_models.py](multilingual_models.py), which loads the dataset, builds a model-specific classifier, trains it with cross-validation, and saves metrics and predictions.

## How the code is organized

- [multilingual_models.py](multilingual_models.py) is the training script. It reads `config.yaml`, loads the data, creates 10 folds, trains the selected model, and writes results.
- [Utils/data_preprocessing.py](Utils/data_preprocessing.py) handles dataset loading and preprocessing for MAMIta, Moxy, and EXIST.
- [Utils/classifiers.py](Utils/classifiers.py) contains the model wrappers, the training loop, the collate function, and the Youden-threshold helper.
- [Utils/results_organizer.py](Utils/results_organizer.py) saves the confusion matrix, classification report, and prediction CSV.

## Configuration

Edit [config.yaml](config.yaml) before running the script.

Important keys:

- `data.dataset_name`: dataset selector such as `MAMIta`, `Moxy`, or `EXIST`.
- `data.data_path`: path to the main dataset file.
- `data.label_path`: path to the label file, if labels are stored separately.
- `data.image_folder`: folder that contains the images.
- `data.label_column`: name of the label column to use.
- `model.text_model_name`: model identifier used to choose the classifier branch.
- `model.processor`: whether the model uses a Hugging Face processor.

## Dataset expectations

- MAMIta: TSV or similar tabular file with meme/text/label information.
- Moxy: Excel workbook with the `Ita` and `Esp` sheets.
- EXIST: JSON or tabular files with meme labels and the corresponding image/text metadata.

## Run

1. Open `config.yaml` and set the dataset paths, image folder, and model name.
2. Make sure the image files referenced by the dataset exist in `data.image_folder`.
3. Run the training script from the repository root:

```bash
python multilingual_models.py
```

## Notes

- The script uses 10-fold cross-validation.
- If `model.threshold` is set to `Youden`, the threshold is estimated from the validation split.
- Results are written to the folder configured in `output.main_output_folder`.
- The current code contains model-specific branches for mCLIP, mBLIP, and multilingual SigLIP.