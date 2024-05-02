<h1 style="font-size:30px; text-align:center"> TRANSVERSE: An any-to-any MultiModal subnet </p>



# Table of Contents
[Introduction](#introduction)

[Key Features](#key-features)

[Roadmap](#roadmap)

[Running miners and validators](#running-miners-and-validators)

[FAQ](#faq)

## Introduction
TransVerse is an any-to-any multimodal model subnet that aims to contribute to the development of AGI(Artificial General Intelligence) through decentralized distributed network in Bittensor ecosystem\
The pursuit of AGI, often referred to as the 'holy grail' of Artificial Intelligence, carries both immense potential and significant challenges in ensuring the alignment of such a powerful system with the well-being of humanity.
Harnessing the unlimited power of incentivized distributed compute resources, TransVerse aspires to build a robust, decentralized, and democratized AGI capable of competing with centralized AI systems.

TransVerse model is the combination of LLM, Image Encoder and Decoder, Video Encoder and Decoder, Audio Encoder and Decoder.
Each modality input is encoded into their own representations, which are then projected to the main LLM.
The LLM acts as the central hub, converting the encoded representations into target modality ones based on text prompts and
These representations are then decoded to the target modality.
The training of TransVerse model is done by training the projection layers of each modality while freezing the LLM, encoders, and decoders of each modality.
This approach offers several benefits:
- Enhanced performance: The use of high-performance encoders and decoders not only improves the quality of the conversion results but also reduces teh overall training scale, making it easier to add additional modalities with minimal effort.
- Scalability: By segregating the modality-specific components (encoders and decoders) from the central LLM, the TransVerse model can be easily expanded to support new modalities without the need to retrain the entire system from scratch.
- Efficient training: The selective training of the projection layers, while keeping the core modality-specific components frozen, significantly reduces the training complexity and resources required, making the model more accessible and easier to deploy.

<img src="docs/imgs/model-structure.png"
alt="Brief structure of Distributed Training"
style="
width: 90%;
padding-left: 5%; 
" />


Here comes the brief structure of the subnet.

Miners:
- Miners are responsible for receiving training batches from the system.
- They compute the gradients on the model based on the received training data.
- Miners then send these computed gradients as responses back to the network.

Validators:
- Validators receive the gradient responses from the miners.
- They validate whether the miners are computing the gradients correctly.
- The validators play a crucial role in ensuring the integrity and reliability of the training process.

Model Server:
- The model server is responsible for aggregating the gradients received from the miners.
- It then propagates the averaged gradients back to the entire system to let miners and validators update their model.

<img src="docs/imgs/distributed-training.png"
alt="Brief structure of Distributed Training"
style="
width: 90%;
padding-left: 5%; 
" />

## Key Features
- Cross-modality Capacity(Text, Image, Video, Audio to Text, Image, Video, and Audio)

- Parameter efficient distributed training through incentivied Bittensor ecosystem

- Combination of high-performing modality-specific models

- Extendability to additional modalities

## Roadmap
- Build incentivized distributed training network to train the any-to-any multimodal model
- Train the state-of-the-art multimodal models by leveraging the unlimited compute resources and open-source datasets
- Host state-of-the-art any-to-any multimodal models
- Add more modalities by combining other modality generative models
- Build multimodal agents based on pretrained models

## Running miners and validators
### Preparation

#### Install the subnet
```
pip install -e .
```

#### Load pretrained model checkpoint from HuggingFace
Create a checkpoint dir
```
mkdir ckpt
mkdir ckpt/pretrained_ckpt
mkdir ckpt/pretrained_ckpt/imagebind_ckpt
mkdir ckpt/pretrained_ckpt/imagebind_ckpt/huge
mkdir ckpt/pretrained_ckpt/7b_tiva_v0
mkdir ckpt/pretrained_ckpt/vicuna_ckpt/
mkdir ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0
```

Download the ImageBind checkpoint
```
wget -P ./ckpt/pretrained_ckpt/imagebind_ckpt/huge https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
```


Download checkpoints from huggingface repository
```
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0
huggingface-cli download 3it/TransVerse-v1 --local-dir ckpt/pretrained_ckpt/7b_tiva_v0
```

### Data Preparation
Simply run the `scripts/data_prep.sh` or follow the instructions to prepare for dataset

---
#### Image Data preparation
```
mkdir data
mkdir data/T-X_pair_data
cd data/T-X_pair_data
huggingface-cli download 3it/TransVerse-Image-Zip --local-dir ./ --repo-type dataset
unzip cc3m.zip
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Image-Zip
rm cc3m.zip
```
#### Video Data preparation
```
huggingface-cli download 3it/TransVerse-Video-Zip --local-dir ./ --repo-type dataset
unzip webvid.zip
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Video-Zip
rm webvid.zip
```
#### Audio Data preparation
```
huggingface-cli download 3it/TransVerse-Audio-Zip --local-dir ./ --repo-type dataset
unzip audiocap.zip
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Audio-Zip
rm audiocap.zip
```

### Running a miner
```
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 neurons/miner.py --subtensor.network test --wallet.name tw --wallet.hotkey tw-h3 --netuid 74 --axon.port 8091 --logging.debug --logging.trace
```

### Running a validator
```
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 neurons/validator.py --subtensor.network test --wallet.name tw --wallet.hotkey tw-h3 --netuid 74 --axon.port 8091 --logging.debug --logging.trace
```


## FAQ
- Which template should I select on runpod?\
Since running the miner and validator code requires PyTorch==1.3.1+cu117, `MiniGPT4 by Camenduru` works fine for them
