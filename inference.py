from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MUSE
import soundfile as sf
import random
h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


# 定义函数来处理声音切片
def process_audio_segment(noisy_wav, model, h, device):
    segment_size = h.segment_size
    n_fft = h.n_fft
    hop_size = h.hop_size
    win_size = h.win_size
    compress_factor = h.compress_factor
    sampling_rate = h.sampling_rate

    # 计算正则化因子
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    orig_size = noisy_wav.size(1)
    # # 判断是否需要补零
    # if noisy_wav.size(1) >= segment_size:
    #     num_segments = noisy_wav.size(1) // segment_size
    #     last_segment_size = noisy_wav.size(1) % segment_size
    #     if last_segment_size > 0:
    #         padded_zeros = torch.zeros(1, segment_size - last_segment_size).to(device)
    #         noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
    #     segments = torch.split(noisy_wav, segment_size, dim=1)
    # 判断是否需要补零
    if noisy_wav.size(1) >= segment_size:
        num_segments = noisy_wav.size(1) // segment_size
        last_segment_size = noisy_wav.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_wav[:, -segment_size:]
            noisy_wav = noisy_wav[:, :-last_segment_size]
            segments = torch.split(noisy_wav, segment_size, dim=1)
            segments = list(segments)
            segments.append(last_segment)
            reshapelast=1
        else:
            segments = torch.split(noisy_wav, segment_size, dim=1)
            reshapelast = 0

    else:
        # 如果语音长度小于一个segment_size，则直接补零
        padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
        noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
        segments = [noisy_wav]
        reshapelast = 0

    # 处理每个语音切片并连接结果
    processed_segments = []
    # for segment in segments:
    #     noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
    #     amp_g, pha_g, com_g = model(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
    #     audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
    #     audio_g = audio_g / norm_factor
    #     processed_segments.append(audio_g.squeeze())
    for i, segment in enumerate(segments):

        noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, com_g = model(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[ :-(segment_size-last_segment_size)]
            # print(orig_size)


        processed_segments.append(audio_g)

    # 将所有处理后的片段连接成一个完整的语音

    processed_audio = torch.cat(processed_segments, dim=-1)
    print(processed_audio.size())
    # 裁切末尾部分，保留noisy_wav长度的部分
    processed_audio = processed_audio[:orig_size]
    print(processed_audio.size())
    print(orig_size)

    return processed_audio


def inference(a):
    model = MUSE(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])



    os.makedirs(a.output_dir, exist_ok=True)
    split=True
    model.eval()

    # 使用torch.no_grad()包裹整个处理过程
    with torch.no_grad():
        for file_name in os.listdir(a.input_noisy_wavs_dir):
            if file_name.lower().endswith('.wav'):
                index = os.path.splitext(file_name)[0]
                print(index)
                noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, file_name), h.sampling_rate)
                noisy_wav = torch.FloatTensor(noisy_wav).to(device, non_blocking=True)
                output_audio = process_audio_segment(noisy_wav, model, h, device)

                output_file = os.path.join(a.output_dir, file_name)
                sf.write(output_file, output_audio.cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_clean_wavs_dir', default='/PATH/TO/YOUR//CLEAN/FILE/VB_DEMAND_16K/clean_train')
    # 选择噪声文件夹
    parser.add_argument('--input_noisy_wavs_dir', default='/PATH/TO/YOUR//NOISY/FILE/VB_DEMAND_16K/noisy_test')
    # parser.add_argument('--input_test_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

