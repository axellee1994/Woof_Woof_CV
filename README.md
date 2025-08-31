# Dog_Vision_CV

End-to-end multi-class dog breed classification using TensorFlow 2.x and TensorFlow Hub. The main analysis and runnable pipeline are in the notebook: [Dog_Vision_CV.ipynb](Dog_Vision_CV.ipynb).

Dataset
- Source: Kaggle / Dog Breed Identification competition
- Archive in this repo: [data/dog-breed-identification.zip](data/dog-breed-identification.zip)
- Notebook expects extracted files (labels.csv, train/, test/) under Google Drive paths as used in the notebook.

Quick results summary (notebook)
- Transfer learning with MobileNetV2 from TensorFlow Hub (see the hub layer usage in the notebook).
- Trained on a small subset first (NUM_IMAGES = 1000) and then on the full set in the notebook.
- Notebook implements training, validation, and final predictions for test/custom images.

Key notebook components and helper functions (openable in the notebook)
- Data and preprocessing:
  - [`get_image`](Dog_Vision_CV.ipynb) — read, decode, normalize, resize images
  - [`get_image_label`](Dog_Vision_CV.ipynb) — build (image, label) tuples
  - [`create_data_batches`](Dog_Vision_CV.ipynb) — build tf.data batches for train/val/test
  - [`show_25_images`](Dog_Vision_CV.ipynb) — visualize a batch of images
- Modeling and training:
  - [`hub_layer_fn`](Dog_Vision_CV.ipynb) — wraps the TF Hub KerasLayer
  - [`create_model`](Dog_Vision_CV.ipynb) — builds the Keras model (hub + Dense)
  - [`train_model`](Dog_Vision_CV.ipynb) — trains with TensorBoard and early stopping
  - [`tensorboard_callback`](Dog_Vision_CV.ipynb) — log directory setup for TensorBoard
- Saving, loading and predictions:
  - [`save_model`](Dog_Vision_CV.ipynb) / [`load_model`](Dog_Vision_CV.ipynb)
  - [`unbatching`](Dog_Vision_CV.ipynb) — convert batched validation set back to arrays
  - [`get_pred_label`](Dog_Vision_CV.ipynb), [`plot_pred`](Dog_Vision_CV.ipynb), [`plot_pred_conf`](Dog_Vision_CV.ipynb)

Techniques demonstrated
- Transfer learning via TensorFlow Hub (MobileNetV2 classification module)
- tf.data pipelines for efficient preprocessing and batching
- TensorBoard logging and EarlyStopping callbacks
- Model export/load with Keras `save` / `load_model`
- Visual prediction inspection and top-N probability plots

Notable files
- Notebook: [Dog_Vision_CV.ipynb](Dog_Vision_CV.ipynb)
- Data archive: [data/dog-breed-identification.zip](data/dog-breed-identification.zip)

Project structure
```
.
├── README.md
├── Dog_Vision_CV.ipynb
└── data/
	└── dog-breed-identification.zip
```

Reproducibility and notes
- Notebook uses Google Drive paths (drive/MyDrive/Dog_Vision/...). Adapt paths if running locally.
- Uses TF Hub model URL configured in the notebook; the hub layer is wrapped by [`hub_layer_fn`](Dog_Vision_CV.ipynb).
- The notebook demonstrates an iterative approach: small-scale experiments (NUM_IMAGES) before full training.

Limitations & next steps
- Small initial training subset leads to overfitting; consider stronger regularization, data augmentation, or larger training runs.
- Replace legacy HDF5 saves with SavedModel or joblib (for non-Keras parts) if needed.
- Improve evaluation (confusion matrix, class-wise metrics) and calibration of probabilities for better deployment.

Open the notebook to run or modify the full pipeline: [Dog_Vision_CV.ipynb](Dog_Vision_CV.ipynb)