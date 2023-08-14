# Fine tune Llama 2 

Scripts for fine-tuning Llama 2 using the Hugging Face TRL library

## Installation dependencies 

### Install pytorch

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Install dependencies
```sh
pip install -U -r requirements.txt
```
### Install protobuf 

If you see the error from the training script. `LlamaConverter requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.`
```sh
pip install protobuf
```
## Hardware requirement

7b and 13b models are able to be SFT and DPO under a single 4090. The 7b model should be able to fit in one 4080 for DPO depending on your LoRa config.

## Fine-tune the model via SFT trainer

```sh
python sft_trainer.py 
```

## Merge the adapter back to the pretrained model

Update the adapter path in `merge_peft_adapters.py` and run the script to merge peft adapters back to pretrained model.
Note that the script is hardcoded to use CPU to merge the model in order to avoid CUDA out of memory errors. However, if you have sufficient VRAM on your GPU, you can change it to use GPU instead.
```sh
python merge_peft_adapters.py
```

##  Fine-tune the model via DPO trainer

```sh
python dpo_trainer.py 
```

## Testing the fine-tuned model.
Update the script `generate.py` and run the script to check the fine-tuned model output.
```sh
python generate.py
```

## Quantization Model
For the 7b or 13b model, because it has the same architecture as the Llama 1 model, you would follow the readme in https://github.com/qwopqwop200/GPTQ-for-LLaMa or https://github.com/PanQiWei/AutoGPTQ. But for 34b or 70b models, you may have to use autoGPTQ.
