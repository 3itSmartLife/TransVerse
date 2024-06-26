
import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import json
import pandas as pd
import bittensor as bt
import torch

# Load a slightly modified version of the Stable Diffusion pipeline.
# This allows us to extract text embeddings directly (without generating images).
from transverse.multimodal.model.custom_sd import StableDiffusionPipeline
from transverse.multimodal.model.custom_vd import TextToVideoSDPipeline
from transverse.multimodal.model.custom_ad import AudioLDMPipeline

class EmbeddingModel(torch.nn.Module):
    "Embedding model"

    def __init__(self, config):
        super(EmbeddingModel, self).__init__()
        self.config = config
        # self.batch_size=config.batch_size
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.audio_pipe = AudioLDMPipeline.from_pretrained(config.model_config['audio_diffusion'], torch_dtype=dtype)
        self.video_pipe = TextToVideoSDPipeline.from_pretrained(config.model_config['video_diffusion'], torch_dtype=dtype)
        self.image_pipe = StableDiffusionPipeline.from_pretrained(config.model_config['image_diffusion'], torch_dtype=dtype)
        
    def forward(self, modality, raw_batch):
        # TODO: Implement embedding extraction mechanism
        caption_list, file_list = raw_batch
        if modality == 'image':
            # TODO: Download the image files in file list
            pass
        elif modality == 'audio':
            # TODO: Download the audio files in file list
            pass
        elif modality == 'video':
            # TODO: Download the video files in file list
            pass
        else:
            # Raise NotImplemented exception
            pass

def save_to_path(emb, path):
    """Save embeddings to disk."""
    try:
        with open(path, 'wb') as wf:
            np.save(wf, emb)
    except:
        bt.logging.error("Error with", path)
    return path


if __name__ == '__main__':

    batch_size = 128

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # clip_output_dir = './embed/'
    # synthesize_path = '../data/synthesize_data/synthesize_data.json'

    # video_path = '../data/T-X_pair_data/webvid/webvid.json'
    # audio_path = '../data/T-X_pair_data/audiocap/audiocap.json'
    # img_path = '../data/T-X_pair_data/cc3m/cc3m.json'

    # image_generation_ckpt_path = 'runwayml/stable-diffusion-v1-5'
    # video_generation_ckpt_path = 'cerspense/zeroscope_v2_576w'
    # audio_generation_ckpt_path = 'cvssp/audioldm-l-full'

    data_path = sys.argv[1]
    modality = sys.argv[2]
    clip_output_dir = sys.argv[3]
    ckpt_path = sys.argv[4]

    if not os.path.exists(clip_output_dir):
        os.makedirs(clip_output_dir, exist_ok=True)

    # Get existing files, so that we don't recompute them.
    existing_files = set([f.strip('.npy') for f in os.listdir(clip_output_dir)])

    caption_list = []
    name_list = []
    if modality == 'audio':
        bt.logging.info('Extract audio caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for row in tqdm(data, total=len(data)):
            one_audio_name, one_caption = row["audio_name"], row["caption"]
            if one_audio_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_audio_name)
        pipe = AudioLDMPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            bt.logging.warning('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'image':
        bt.logging.info('Extract image caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            one_image_name, one_caption = row["image_name"], row["caption"]
            if one_image_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_image_name)
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            bt.logging.warning('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'video':
        bt.logging.info('Extract video caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            one_video_name, one_caption = row["video_name"], row["caption"]
            if one_video_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_video_name)
        pipe = TextToVideoSDPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            bt.logging.warning('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")

    bt.logging.info('Extract embeddings in batches.')
    num_batches = int(np.ceil(len(caption_list) / batch_size))
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_captions = caption_list[start_idx:end_idx]
        batch_ids = name_list[start_idx:end_idx]
        prompt_embeds = pipe(batch_captions, return_prompts_only=True).detach().cpu().numpy()

        # Save embeddings to disk in parallel.
        Parallel(n_jobs=8)(delayed(save_to_path)(
            prompt_embeds[j, :, ...], os.path.join(clip_output_dir, f'{batch_ids[j]}.npy')
        ) for j in range(prompt_embeds.shape[0]))
