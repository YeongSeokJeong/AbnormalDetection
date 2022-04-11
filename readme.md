# AbnormalDetection

## environment installation

### create env file
- You must create `.env` file to set environmental variables.
```
wandb_api_key=[Your Key] # "xxxxxxxxxxxxxxxxxxxxxxxx"
data_dir=[Your Path] # "/home/repos/open"
```
- `wandb_api_key` is able to get from [this link](https://wandb.ai/authorize).
- `data_dir` is the directory name where datafile is downloaded.

### conda installation
```angular2html
conda env create -f environment.yaml
conda activate abnormal
```

## train model
```angular2html
python main.py
```
- If you want to run pretrained `resnet50`, you simply run above.
- If you want to change the configuration, you need to make new yaml file in the `config` directory.
