# MUSE: Flexible Voiceprint Receptive Fields and Multi-Path Fusion Enhanced Taylor Transformer for U-Net-based Speech Enhancemen
### Zizhen Lin, Xiaoting Chen, Junyu Wang

**Abstract:** 
 Achieving a balance between lightweight design and high performance remains a challenging task for speech enhancement. In this paper, we introduce Multi-path Enhanced Taylor (MET) Transformer based U-net for Speech Enhancement (MUSE), a lightweight speech enhancement network built upon the U-net architecture. Our approach incorporates a novel Multi-path Enhanced Taylor (MET) Transformer block, which integrates Deformable Embedding (DE) to enable flexible receptive fields for voiceprints. The MET Transformer is uniquely designed to fuse Channel and Spatial Attention (CSA) branches, facilitating channel information exchange and addressing spatial attention deficits within the Taylor-Transformer framework. Through extensive experiments conducted on the VoiceBank+DEMAND dataset, we demonstrate that MUSE achieves competitive performance while significantly reducing both training and deployment costs, boasting a mere 0.51M parameters.


## Pre-requisites
1. Python >= 3.6.
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](https://github.com/yxlu-0102/MP-SENet/blob/main/requirements.txt).
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942). Use downsampling.py to resample all wav files to 16kHz, 
```
python downsampling.py
```
5. move the clean and noisy wavs to `VoiceBank+DEMAND/wavs_clean` and `VoiceBank+DEMAND/wavs_noisy` or any path you want, and change the path in train.py [parser.add_argument('--input_clean_wavs_dir', default=], respectively. Notably, different downsampling ways could lead to different result. 

## Training
For single GPU (Recommend), MUSE needs at least 8GB GPU memery.
```
python train.py --config config.json
```

## Training with your own data

Edit path in make_file_list.py and run

```
python make_file_list.py
```
Then replace the test.txt and training.txt with generated files in folder ./VoiceBank+DEMAND and put your train and test set in the same folder(clean, noisy).

## Inference
```
python inference.py --checkpoint_file /PATH/TO/YOUR/CHECK_POINT/g_xxxxxxx
```
You can also use the pretrained best checkpoint file we provide in `best_ckpt/g_best`.<br>
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MB-TaylorFormer](https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer)
