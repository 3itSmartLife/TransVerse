# Data Preparation

### Image Data preparation
huggingface-cli download 3it/TransVerse-Image-Zip --local-dir ./ --repo-type dataset
zip -FF cc3m.zip --out cc3m-full.zip
rm cc3m.z*
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Image-Zip/
unzip cc3m-full.zip
rm cc3m-full.zip

### Video Data preparation
huggingface-cli download 3it/TransVerse-Video-Zip --local-dir ./ --repo-type dataset
zip -FF webvid.zip --out webvid-full.zip
rm webvid.z*
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Video-Zip/
unzip webvid-full.zip
rm webvid-full.zip

### Audio Data preparation
huggingface-cli download 3it/TransVerse-Audio-Zip --local-dir ./ --repo-type dataset
zip -FF audiocap.zip --out audiocap-full.zip
rm audiocap.z*
rm -rf ~/.cache/huggingface/hub/datasets--3it--TransVerse-Audio-Zip/
unzip audiocap-full.zip
rm audiocap-full.zip
