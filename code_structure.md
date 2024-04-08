
### 1. Code Structure 
```
├── neurons
│   ├── miner.py
│   │   ├── Miner(Class)
│   │   │   ├── __init__()
│   │   │   │   ├── Load the model
│   │   │   ├── forward()
│   │   │   │   ├── Get synapse from the validator
│   │   │   │   ├── Calculate the gradient
│   │   │   │   ├── Respond to the validator through synapse.gradients
│   │   │   ├── blacklist()
│   │   │   │   ├── Blacklist validators who are not registered
│   │   │   │   ├── Blacklist validators without enough stake
│   │   │   └── priority()
│   │   │   │   ├── Prioritize validators according to workloads
│   │   │   │   ├── Prioritize validators according to their stake amount
│   ├── validator.py
│   │   ├── Validator(Class)
│   │   │   ├── __init__()
│   │   │   │   ├── Load the model
│   │   │   │   ├── Load the dataset
│   │   │   ├── forward()
│   │   │   │   ├── TODO
├── transverse
│   ├── api
│   │   │   ├── __init__.py
│   │   │   │   ├── TODO: Confirm the functionality
│   │   │   ├── get_query_axons.py
│   │   │   │   ├── TODO: Confirm the functionality
│   │   │   ├── transverse_api.py
│   │   │   │   ├── TODO: Confirm the functionality
│   ├── base                            # Little to do with default validator and miner
│   │   │   ├── __init__.py
│   │   │   ├── neuron.py
│   │   │   │   ├── BaseNueron(Class)
│   │   │   │   │   ├── TODO: Confirm the functionality
│   │   │   ├── miner.py
│   │   │   │   ├── BaseMinerNeuron(Class)
│   │   │   │   │   ├── __init__()
│   │   │   │   │   ├── run()
│   │   │   │   │   ├── run_in_background_thread()
│   │   │   │   │   ├── stop_run_thread()
│   │   │   │   │   ├── __enter__()
│   │   │   │   │   ├── __exit__()
│   │   │   │   │   ├── resync_metagraph()
│   │   │   ├── validator.py
│   │   │   │   ├── BaseValidatorNeuron(Class)
│   │   │   │   │   ├── add_args()
│   │   │   │   │   ├── __init__()
│   │   │   │   │   ├── serve_axon()
│   │   │   │   │   ├── concurrent_forward()
│   │   │   │   │   ├── run()
│   │   │   │   │   ├── run_in_background_thread()
│   │   │   │   │   ├── stop_run_thread()
│   │   │   │   │   ├── __enter__()
│   │   │   │   │   ├── __exit__()
│   │   │   │   │   ├── set_weights()
│   │   │   │   │   ├── resync_metagraph()
│   │   │   │   │   ├── update_scores()
│   │   │   │   │   ├── save_state()
│   │   │   │   │   ├── load_state()
│   ├── miner
│   ├── validator
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   │   ├── load_dataset(Function)
│   │   │   │   ├── TODO: Implement dataset loading mechanism
│   │   │   ├── random_batch(Function)
│   │   │   │   ├── TODO: Implement random batch selection mechanism
│   │   │   ├── extract_embeddings(Function)
│   │   │   │   ├── TODO: Implement embedding extraction mechanism
│   │   ├── forward.py
│   │   │   ├── forward(Function)
│   │   │   │   ├── TODO: Implement random miner selection mechanism
│   │   │   │   ├── TODO: Adjust random_batch function call
│   │   │   │   ├── TODO: Adjust extract_embeddings function call
│   │   │   │   ├── TODO: Implement verification mechanism
│   │   │   │   ├── TODO: Implement rwarding function call
│   │   ├── reward.py
│   │   │   ├── reward(Function)
│   │   │   │   ├── TODO: Implement rewarding mechanism
│   │   │   ├── get_rewards(Function)
│   │   │   │   ├── TODO: 
│   │   ├── sync_model.py
│   │   │   ├── sync_model(Function)
│   │   │   │   ├── TODO: Implement model sync mechanism
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config.py
│   │   │   ├── TODO: Check and add configs
│   │   ├── misc.py
│   │   │   ├── DONE: Already done by default
│   │   ├── uids.py
│   │   │   ├── check_uid_availability(Function)
│   │   │   │   ├── TODO: Check the function implementation again
│   │   │   ├── get_random_uids(Function)
│   │   │   │   ├── DONE: Already done by default
│   ├── model
│   ├── protocol.py (Consider moving into a dir)
│   │   ├── DistributedTraining(Class)
│   │   │   ├── deserialize(Function)
│   ├── mock.py (Consider moving into a dir)
├── LICENCE.md
├── README.md
└── requirements.txt
```












```
├── figures
├── data
│   ├── T-X_pair_data  
│   │   ├── audiocap                      # text-autio pairs data
│   │   │   ├── audios                    # audio files
│   │   │   └── audiocap.json             # the audio captions
│   │   ├── cc3m                          # text-image paris data
│   │   │   ├── images                    # image files
│   │   │   └── cc3m.json                 # the image captions
│   │   └── webvid                        # text-video pairs data
│   │   │   ├── videos                    # video files
│   │   │   └── webvid.json               # the video captions
│   ├── IT_data                           # instruction data
│   │   ├── T+X-T_data                    # text+[image/audio/video] to text instruction data
│   │   │   ├── alpaca                    # textual instruction data
│   │   │   ├── llava                     # visual instruction data
│   │   ├── T-T+X                         # synthesized text to text+[image/audio/video] instruction data
│   │   └── MosIT                         # Modality-switching Instruction Tuning instruction data
├── code
│   ├── config
│   │   ├── base.yaml                     # the model configuration 
│   │   ├── stage_1.yaml                  # enc-side alignment training configuration
│   │   ├── stage_2.yaml                  # dec-side alignment training configuration
│   │   └── stage_3.yaml                  # instruction-tuning configuration
│   ├── dsconfig
│   │   ├── stage_1.json                  # deepspeed configuration for enc-side alignment training
│   │   ├── stage_2.json                  # deepspeed configuration for dec-side alignment training
│   │   └── stage_3.json                  # deepspeed configuration for instruction-tuning training
│   ├── datast
│   │   ├── base_dataset.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── cc3m_datast.py                # process and load text-image pair dataset
│   │   ├── audiocap_datast.py            # process and load text-audio pair dataset
│   │   ├── webvid_dataset.py             # process and load text-video pair dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   ├── T-T+X_instruction_dataset.py  # process and load text-to-text+x instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model                     
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├── common
│   │   ├── anyToImageVideoAudio.py       # the main model file
│   │   ├── agent.py
│   │   ├── modeling_llama.py
│   │   ├── custom_ad.py                  # the audio diffusion 
│   │   ├── custom_sd.py                  # the image diffusion
│   │   ├── custom_vd.py                  # the video diffusion
│   │   ├── layers.py                     # the output projection layers
│   │   └── ...  
│   ├── scripts
│   │   ├── train.sh                      # training NExT-GPT script
│   │   └── app.sh                        # deploying demo script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   ├── inference.py                      # inference
│   ├── demo_app.py                       # deploy Gradio demonstration 
│   └── ...
├── ckpt                           
│   ├── delta_ckpt                        # tunable NExT-GPT params
│   │   ├── nextgpt         
│   │   │   ├── 7b_tiva_v0                # the directory to save the log file
│   │   │   │   ├── log                   # the logs
│   └── ...       
│   ├── pretrained_ckpt                   # frozen params of pretrained modules
│   │   ├── imagebind_ckpt
│   │   │   ├──huge                       # version
│   │   │   │   └──imagebind_huge.pth
│   │   ├── vicuna_ckpt
│   │   │   ├── 7b_v0                     # version
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   │   ├── tokenizer.model
│   │   │   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```

