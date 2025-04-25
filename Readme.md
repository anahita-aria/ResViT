# Improved Deepfake Detection Using ResNet + Vision Transformer (ResViT)

This repository is based on [CViT](https://github.com/erprogs/CViT) and extends it by replacing the custom CNN feature extractor with modified **ResNet-50**, enhancing the model’s ability to detect frame-level deepfakes. The ResNet backbone improves feature quality and model generalization while maintaining computational efficiency.

---

##  Differences from CViT

- **Feature Extractor**:  
  Replaced custom CNN with modified **ResNet-50** to extract more robust and hierarchical features from input frames.

- **Planned Enhancements – TokenLearner (Future Work)**:  
  TokenLearner is planned to be integrated in future versions to reduce the Vision Transformer’s complexity. It selects only the most informative tokens, improving training speed and memory usage while maintaining prediction accuracy.

---

## Requirements

- Python 3.7+
- PyTorch ≥ 1.10
- FFmpeg
- Install packages:
  ```bash
  pip install GPUtil einops face_recognition facenet-pytorch dlib
  conda install -y -c conda-forge ncurses
  ```

---

##  Face Extraction Components

The following tools and scripts are used for face extraction and video preprocessing:

- `helpers_read_video_1.py`
- `helpers_face_extract_1.py`
- `blazeface.py`
- `blazeface.pth` (pretrained BlazeFace weights)
- `face_recognition`
- `facenet-pytorch`
- `dlib`

---

##  Preprocessing

### 1. Extract Video Frames

Use `ffmpeg` to extract frames from original videos and save them at `224x224` resolution. You can change the frame rate (`fps=3`) based on your storage capacity.

```bash
# Real frames
for f in <Your_path_to_the_dataset>/*; do \
  ffmpeg -hide_banner -loglevel quiet -i "$f" \
  -vf "fps=3,scale=224:224" \
  <Your_destination_path>/OriginalFrames/$(basename "$f" .mp4)_%0d.png ; \
done

# Fake frames
for f in <Your_path_to_the_dataset>/*; do \
  ffmpeg -hide_banner -loglevel quiet -i "$f" \
  -vf "fps=3,scale=224:224" \
  <Your_destination_path>/DeepfakesFrames/$(basename "$f" .mp4)_%0d.png ; \
done
```

### 2. Organize Frame Folders

After extracting frames, split them based on metadata (from your dataset, e.g., DFDC or FaceForensics++) into:


```
/DatasetRoot/
├── OriginalFrames/
│   ├── Train/
│   │   ├── Real/
│   │   └── Fake/
│   ├── Validation/
│   │   ├── Real/
│   │   └── Fake/
│   └── Test/
│       ├── Real/
│       └── Fake/
├── DeepfakesFrames/
│   ├── Train/
│   │   ├── Real/
│   │   └── Fake/
│   ├── Validation/
│   │   ├── Real/
│   │   └── Fake/
│   └── Test/
│       ├── Real/
│       └── Fake/
```

- `OriginalFrames` should contain extracted frames from authentic videos.
- `DeepfakesFrames` should contain frames from manipulated (fake) videos.
- Distribute these frames into `Train`, `Validation`, and `Test` folders by referencing metadata provided by the dataset you are using.

This structure is essential for training the model properly and for the loader scripts to function as expected.

---

##  Train ResViT

```bash
python ResViT/ResViT_train_model2.py \
  -e 50 \
  -s 'g' \
  -l 0.0001 \
  -w 0.0000001 \
  -d /path/to/DatasetRoot/ \
  -b 32
```

**Arguments**:
- `-e`: number of epochs
- `-s`: model selector ('g' uses ResNet50-based structure)
- `-l`: learning rate
- `-w`: weight decay
- `-d`: dataset root directory
- `-b`: batch size

---

##  Prediction

```bash
python ResViT_prediction_model2.py \
  --p <video path> \
  --f <number_of_frames> \
  --d <dataset_type> \
  --w <weights_path> \
  --n <network_type> \
  --fp16 <half_precision>
```

### Output:
- Prediction < 0.5 → **REAL**
- Prediction ≥ 0.5 → **FAKE**

### Argument Details:

- `--p (str)`: Path to video or image file.  
  *Example*: `/path/to/video.mp4`

- `--f (int)`: Number of frames to process.  
  *Example*: `30`

- `--d (str)`: Dataset type: `dfdc`, `faceforensics`, `timit`, or `celeb`.

- `--w (str)`: Path to model weights.  
  *Example*: `weight/resvit_ep_50.pth`

- `--n (str)`: Network type.  
  *Example*: `resvit`


---

##  Datasets

This work is tested using:

- [DFDC - Deepfake Detection Challenge](https://ai.facebook.com/datasets/dfdc)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)

---

## Repository Structure

```
ResViT/
├── model/                    # ResNet-50 + Vision Transformer
├── helpers/                  # Training and evaluation logic
├── preprocessing/            # Frame extraction and setup
├── weight/                   # Trained model weights
├── ResViT_train_model2.py    # Training entry point
├── ResViT_prediction_model2.py # Prediction entry point
```

---

##  Notes

- This project replaces a custom CNN with a modified ResNet-50 for better feature representation.
- Future updates will integrate TokenLearner to improve transformer efficiency.
- Frame-level classification can be extended to video-level aggregation.

---

##  Author

Anahita Aria  
This project is intended for academic and research use.

---

##  Citation

If you use this work, please cite it using the following BibTeX:

```bibtex
@misc{aria2025resvit,
  author       = {Anahita Aria},
  title        = {ResViT: Improved Deepfake Detection using ResNet and Vision Transformer},
  year         = {2025},
  howpublished = {\url{https://github.com/anahita-aria/ResViT}},
  note         = {Forked and extended from CViT: https://github.com/erprogs/CViT}
}
```
