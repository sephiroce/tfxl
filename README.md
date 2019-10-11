# Transformer-XL
* This repository is a customized version of the original Transformer-XL source code (https://github.com/kimiyoung/transformer-xl)
* Reference results: PPL 54.5 @ PTB
* It only supports Tensorflow, Single machine, Multi-GPUs

## Environment
* Python : 3.6.4
* Tensorflow: 1.14

## Guide to run Transformer-XL (from the original repository)

### (1) Preprocess raw data and create tfrecords
* create training and validation data: `bash scripts/dataset_bas_gpu.sh train_data`
* create test data: `bash scripts/dataset_base_gpu.sh test_data`

### (2) Run training
* Modify the configurations in `scripts/dataset_base_gpu.sh`  according to your needs.
* `bash scripts/dataset_base_gpu.sh train`
* If enough resources are available, increasing the model sizes (e.g., `N_LAYER`, `D_MODEL`, `D_EMBED`, `D_HEAD`, `D_INNER`) so that they are closer to the values defined in `scripts/dataset_large_tpu.sh`. Likewise, when resources are limited, decrease the model sizes. It is recommended to ensure that `D_MODEL == D_EMBED` and `D_MODEL == N_HEAD x D_HEAD`. When the model sizes increase, remember to increase `warmup_steps` accordingly to alleviate optimization difficulties.
* Adjust the `NUM_CORE` parameter to reflect the number of GPUs to use.

### (3) Run evaluation
* `bash scripts/dataset_base_gpu.sh eval --eval_ckpt_path PATH_TO_CKPT`