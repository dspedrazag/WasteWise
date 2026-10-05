# WasteWise

WasteWise is a prototype AI-assisted recycling tool designed to provide
Gainesville-specific disposal guidance from an image of a waste item.

This repository currently contains the initial dataset exploration for the project.

## Dataset

The current exploratory dataset is **TrashNet**, created by Gary Thung and
Mindy Yang.

TrashNet contains 2,527 images divided into six classes:

- cardboard
- glass
- metal
- paper
- plastic
- trash

The dataset is used for initial computer-vision experiments and data exploration.

Source:
https://github.com/garythung/trashnet

Dataset mirror:
https://huggingface.co/datasets/garythung/trashnet

The author's Hugging Face repository identifies the dataset with an MIT license.

## Dataset storage

The dataset files are not committed to this GitHub repository.

The `data/` directory is excluded using `.gitignore`.

The notebook can download `dataset-resized.zip` from the public TrashNet
repository if the dataset is not already available locally.

## Dataset exploration

The main exploratory notebook is:

`playground.ipynb`

The notebook currently:

- loads and extracts the dataset
- counts images by class
- visualizes class distribution
- displays example images
- checks image formats and dimensions
- checks for unreadable images
- identifies limitations relevant to WasteWise

## Initial observations

TrashNet is not perfectly balanced. The `trash` class contains substantially
fewer images than the other categories.

All 2,527 images are JPEG files with a resolution of 512 × 384 pixels.

TrashNet categories are also broader than the target WasteWise categories.
For example, `plastic` is not limited to plastic bottles and `metal` is not
limited to aluminum cans. A more task-specific dataset or additional filtering
may therefore be needed later.

## Environment

Install the required Python packages with:

```bash
pip install -r requirements.txt