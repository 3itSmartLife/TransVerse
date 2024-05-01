#!/bin/bash

mkdir data
mkdir data/T-X_pair_data
cd data/T-X_pair_data

# Download image dataset
huggingface-cli download 3it/TransVerse-Image-Zip --local-dir ./ --repo-type dataset
unzip cc3m.zip
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Image-Zip
rm cc3m.zip

# Download video dataset
huggingface-cli download 3it/TransVerse-Video-Zip --local-dir ./ --repo-type dataset
unzip webvid.zip
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Video-Zip
rm webvid.zip

# Download audio dataset
huggingface-cli download 3it/TransVerse-Audio-Zip --local-dir ./ --repo-type dataset
unzip audiocap.zip
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Audio-Zip
rm audiocap.zip