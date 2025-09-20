**Automatic Chord Recognition Using Deep Learning Techniques**



This repository implements a Transformer based model for **automatic chord recognition**. 

 

The pipeline converts audio into constant-Q transform (CQT) features and predicts chord sequences using a Transformer-inspired model.



---



## Project Structure

```

BTC\_Model/
├── models/ # Model architectures (baseline, CRF, Transformer, etc.)
├── utils/ # Utilities: preprocessing, metrics, logging, etc.
│ ├── preprocess.py
│ ├── chords.py
│ ├── pytorch\_utils.py
│ └── ...
├── audio\_dataset.py # Dataset loader
├── train.py # Training script
├── train\_crf.py # Training with CRF
├── test.py # Evaluation script
└── requirements.txt # Python dependencies

```

---



## Features

- CQT-based preprocessing with librosa
- Large-vocabulary support (maj/min + extended chords, ~170 classes)
- Evaluation using MIREX metrics and Weighted Chord Symbol Recall (WCSR)  
- Modular design for experiments and easy extension

---


## Getting Started



### 1. Clone this repository
```bash

git clone https://github.com/rocelload/Automatic-Chord-Recognition-Using-Deep-Learning-Techniques.git

cd Automatic-Chord-Recognition-Using-Deep-Learning-Techniques

```



### 2. Set up a virtual environment

```

python -m venv .venv

# On Windows

.venv\\Scripts\\activate

# On Linux/Mac

source .venv/bin/activate

```



### 3. Install dependencies
```

pip install -r requirements.txt

```



### Training Example
```

python train.py --config configs/cnn\_baseline.yaml

python train\_crf.py --config configs/crf.yaml

python test.py --ckpt checkpoints/model.ckpt --dataset isophonics
```



### Evaluation

```

python test.py --ckpt checkpoints/model.ckpt --dataset isophonics

```


### References



This work is inspired by prior research in chord recognition and sequence modeling:

- Sigtia et al. (2015) – Hybrid RNN for Audio Chord Recognition
- McFee \& Bello (2017) – Structured Training for Large-Vocabulary Chord Recognition
- Humphrey \& Bello (2012) – Rethinking Automatic Chord Recognition with CNNs
- Vaswani et al. (2017) – Attention Is All You Need
- BTC-ISMIR19 Repository – jayg996/BTC-ISMIR19
---





