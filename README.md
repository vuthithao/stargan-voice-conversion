# StarGAN-Voice-Conversion
This is a pytorch implementation of the paper: StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks  https://arxiv.org/abs/1806.02169 .
Note that the model architecture is a little different from that of the original paper.

# Dependencies
* Python 3.6 (or 3.5)
* Pytorch 0.4.0
* pyworld
* tqdm
* librosa
* tensorboardX and tensorboard

or
```
conda env create -f environment.yml
```

# Usage
## Download Dataset

Download and unzip [data and trained model](https://drive.google.com/drive/folders/1kQE0rg-y2YLHSYaiwMqdEYoxZvrOLy0B?usp=sharing) corpus to designated directories.

If you want make your data (anyone you want), Download audiobook [here](http://sachnoiviet.com/trang-chu.html)
Then follow the steps in `data_generator.ipynb`

Preprocess data

We will use Mel-cepstral coefficients(MCEPs) here.

```bash
python preprocess.py
```

Train model

Note: you may need to early stop the training process if the training-time test samples sounds good or the you can also see the training loss curves to determine early stop or not.

```
python main.py
```
Trained model [here](https://drive.google.com/drive/folders/1kQE0rg-y2YLHSYaiwMqdEYoxZvrOLy0B?usp=sharing)

Convert

For example: restore model at step 200000 and specify the source speaker and target speaker to `p0` and `p7`, respectively.

```
python convert.py --resume_iters 200000 --src_spk p0 --trg_spk p7 --filename ./data/sontung/test/p0_691_63_2.wav
```
Post-process: follow `denoise.ipynb`

Demo
```
cd demo_audio
python server.py
```
`103.137.4:3003`

API
```
python api.py
```

Post request sample
```
python post_request.py
```

## To-Do list
- [x] Post some converted samples (Please find some converted samples in the `converted_samples` folder).

## Papers that use this repo:
1. [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss (ICML2019)](https://arxiv.org/pdf/1905.05879v2.pdf)
2. [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion (NeurIPS 2019)](https://arxiv.org/pdf/1906.00794.pdf)
3. [ADAGAN: ADAPTIVE GAN FOR MANY-TO-MANY NON-PARALLEL VOICE CONVERSION (under review for ICLR 2020)](https://openreview.net/pdf?id=HJlk-eHFwH)

