# Transformer-XL
* This repository is a customized version of the original Transformer-XL source code (https://github.com/kimiyoung/transformer-xl)
* Reference results: PPL 54.5 @ PTB
* It only supports Tensorflow, Single machine, Multi-GPUs

## Environment
* Python : 3.6.4
* Tensorflow: 1.14

## Guide to run Transformer-XL with your own text corpus.
### Before training
* preprocess your text corpus by yourself. 
* Refer to the paper to set up the model architecture. (You can find these options in ```ptb_run.sh``` which is a sample bash script.)
```
# base model settings for one-billion corpus
 
# Model
DIV_VAL=4
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=8
D_HEAD=128
D_INNER=4096

# Training
TGT_LEN=256
MEM_LEN=256

BSZ=256
NUM_CORE=4

# Testing
TEST_TGT_LEN=32
TEST_MEM_LEN=128
TEST_CLAMP_LEN=-1

TEST_BSZ=16
TEST_NUM_CORE=1
```

### Building up vocabulary
* Make vocab file: You can simply build up your own vocabulary file as shown below.
```
sed -e 's/ /\n/g' ${TRAIN_CORPUS} | sort -u > ${VOCAB_FILE}
```
* You can add special symbols such as "\<s\>", "\</s\>", and "\<unk\>".
* The word index is set to ```${row_number} - 1```.

### Creating tfrecords
* We need to use different target lengths and batch sizes depending on the type of the text corpus. 
#### Train & Valid corpus
* put all the text files into data/${data_name}/train
* merge your valid text files into data/${data_name}/valid.txt
```
python3 tfxl/data_utils.py \
    --data_dir=${DATA_ROOT}/ \
    --vocab=${VOCAB_FILE} \
    --dataset=${NAME} \
    --tgt_len=${TGT_LEN} \
    --per_host_train_bsz=${BSZ} \
    --per_host_valid_bsz=${BSZ}
```
#### Test corpus
* merge your test text files into data/${data_name}/test.txt
```
python tfxl/data_utils.py \
    --data_dir=${DATA_ROOT}/ \
    --vocab=${VOCAB_FILE} \
    --dataset=${NAME} \
    --tgt_len=${TEST_TGT_LEN} \
    --per_host_test_bsz=${TEST_BSZ}
```

### Run training  (Guides from the original repository)
* If enough resources are available, increasing the model sizes (e.g., `N_LAYER`, `D_MODEL`, `D_EMBED`, `D_HEAD`, `D_INNER`) so that they are closer to the values defined in `scripts/dataset_large_tpu.sh`. Likewise, when resources are limited, decrease the model sizes. It is recommended to ensure that `D_MODEL == D_EMBED` and `D_MODEL == N_HEAD x D_HEAD`.
* When the model sizes increase, remember to increase `warmup_steps` accordingly to alleviate optimization difficulties.
* Adjust the `NUM_CORE` parameter to reflect the number of GPUs to use.  
```
python tfxl/train_gpu.py \
    --data_dir=${DATA_ROOT}/tfrecords \
    --vocab=${VOCAB_FILE} \
    --record_info_dir=${DATA_ROOT}/tfrecords/ \
    --corpus_info_path=${DATA_ROOT}/corpus-info.json \
    --model_dir=EXP-${NAME} \
    --n_layer=${N_LAYER} \
    --d_model=${D_MODEL} \
    --d_embed=${D_EMBED} \
    --n_head=${N_HEAD} \
    --d_head=${D_HEAD} \
    --d_inner=${D_INNER} \
    --dropout=0.1 \
    --dropatt=0.0 \
    --learning_rate=0.00025 \
    --warmup_steps=0 \
    --train_steps=400000 \
    --tgt_len=${TGT_LEN} \
    --mem_len=${MEM_LEN} \
    --train_batch_size=${BSZ} \
    --iterations=200 \
    --save_steps=4000 \
    --do_train=True \
    --do_eval=False \
```

### Run evaluation
```
python tfxl/train_gpu.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --vocab=${VOCAB_FILE} \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=EXP-${NAME} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test
```