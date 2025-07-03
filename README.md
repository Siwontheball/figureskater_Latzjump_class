# Figure Skate Analysis

# This repository provides an end-to-end pipeline to classify jump types in real-time, overlaying these annotations on the output video. I used an LSTM model trained on 2D skeleton sequences. I used the source of the dataset is https://www.kaggle.com/datasets/betessawildenboer/lutz-jumps-dataset?resource=download but you can train the model with your own dataset. The validation score I got was around 70%, and I recommend that you use the trainset with the clear resolutions of ankles.

Repository Structure
figure_skate_analysis/
├─ data/                  # (not tracked)
│  └─ skeletons/          # training `.npy` skeletons from nightsky37
│  └─ labels.csv          # CSV mapping skeleton filenames to jump types
│
├─ input/
│  └─ full_video.mp4      # your raw skating video
│
├─ models/
│  └─ best_model.pt       # trained LSTM jump-classifier weights
│
├─ src/
│  ├─ main.py             # runs YOLO → velocity → pose → classification → annotation
│  ├─ dataset.py          # PyTorch Dataset for skeleton training data
│  ├─ model.py            # LSTMClassifier definition
│  ├─ train.py            # train & validate the jump-classifier, saves best_model.pt
│  ├─ extract_skeletons.py# wrapper for running pose estimator on clips
│  ├─ pose.py             # extract_keypoints() wrapper (MediaPipe/OpenPose)
│  └─ config.yaml         # paths & hyperparameters
│
├─ output/
│  └─ annotated.mp4       # final annotated video (generated)
│
└─ requirements.txt       # Python dependencies


#Setup & Installation

1. Clone this repository:
git clone https://github.com/your-username/figure_skate_analysis.git
cd figure_skate_analysis

2. Create a virtual environment and install dependencies:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3. Populate folders:
Place your raw video in input/full_video.mp4.
Place training skeletons in data/Flat/, data/Inside, and data/Right and data/processed/.
If you already have pre-trained jump-classifier weights, place that as a models/best_model.pt.

4. Training the Jump Classifier
If you need to train or retrain the jump-classifier:
cd src
python train.py
Training logs and best weights will be saved under models/best_model.pt.
Adjust hyperparameters in src/config.yaml as needed.

5. Training logs and best weights will be saved under models/best_model.pt.
Adjust hyperparameters in src/config.yaml as needed.
cd src
python main.py
The annotated video will be saved to output/annotated.mp4.
Key settings (FPS, buffer length, thresholds) can be tuned in src/config.yaml.