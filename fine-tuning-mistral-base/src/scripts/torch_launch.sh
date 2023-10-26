#!/bin/bash
WORKING_DIR=/mnt/c/Users/USER/PycharmProjects/krikri-sagemaker-test/fine-tuning-mistral-base/src/scripts
SM_WORKING_DIR=/mnt/c/Users/USER/PycharmProjects/krikri-sagemaker-test/fine-tuning-mistral-base/model

#The related information about multi-nodes cluster.
# MASTER_HOST=$SM_MASTER
MASTER_ADDR="localhost"
MASTER_PORT="9994"
NNODES="1"
NODE_RANK="0"

GPUS_PER_NODE="1"
DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/ds_config.json"
SEED=42
EPOCHS=1
TOKEN=$HF_TOKEN
model_id="mistralai/Mistral-7B-v0.1"

train_dataset_path='/mnt/f/processed/mistral/dolly/train'
# test_dataset_path='/opt/ml/input/data/test'
learning_rate=0.00001
max_grad_norm=0.3
warmup_ratio=0.03
per_device_train_batch_size=6
# per_device_eval_batch_size=1
gradient_accumulation_steps=2
lr_scheduler_type="constant_with_warmup"
save_strategy="epoch"
logging_steps=10

OPTS=""
# OPTS+=" --per_device_eval_batch_size ${per_device_eval_batch_size}"
OPTS+=" --model_id ${model_id}"
OPTS+=" --per_device_train_batch_size ${per_device_train_batch_size}"
OPTS+=" --gradient_accumulation_steps ${gradient_accumulation_steps}"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --bf16 True"
OPTS+=" --tf32 True"
OPTS+=" --max_grad_norm ${max_grad_norm}"
OPTS+=" --warmup_ratio ${warmup_ratio}"
OPTS+=" --lr_scheduler_type ${lr_scheduler_type}"
OPTS+=" --learning_rate ${learning_rate}"
OPTS+=" --dataset_path ${train_dataset_path}"
# OPTS+=" --test_dir ${test_dataset_path}"
OPTS+=" --deepspeed ${DS_CONFIG}"
OPTS+=" --num_train_epochs ${EPOCHS}"
OPTS+=" --logging_steps ${logging_steps}"
OPTS+=" --merge_adapters True"
OPTS+=" --use_flash_attn True"
OPTS+=" --output_dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"


CMD="deepspeed  ${DISTRIBUTED_ARGS}  ${WORKING_DIR}/meltemi_finetuning.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log