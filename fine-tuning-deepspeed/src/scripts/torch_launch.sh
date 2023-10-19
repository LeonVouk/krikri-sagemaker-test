#!/bin/bash
WORKING_DIR=/opt/ml/code
SM_WORKING_DIR=/opt/ml/model

#The related information about multi-nodes cluster.
MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"

GPUS_PER_NODE="$SM_NUM_GPUS"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

SAVE_PATH="${SM_WORKING_DIR}/results"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/ds_config.json"
SEED=42
EPOCHS=1
TOKEN=$HF_TOKEN
model_id="meta-llama/llama-2-7b-hf"

train_dataset_path='/opt/ml/input/data/train'
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
OPTS+=" --per_device_train_batch_size ${per_device_train_batch_size}"
OPTS+=" --gradient_accumulation_steps ${gradient_accumulation_steps}"
OPTS+=" --gradient_checkpointing"
OPTS+=" --bf16"
OPTS+=" --tf32"
OPTS+=" --max_grad_norm ${max_grad_norm}"
OPTS+=" --warmup_ratio ${warmup_ratio}"
OPTS+=" --lr_scheduler_type ${lr_scheduler_type}"
OPTS+=" --model_id ${model_id}"
OPTS+=" --distributed_backend nccl"
OPTS+=" --learning_rate ${learning_rate}"
OPTS+=" --dataset_path ${train_dataset_path}"
# OPTS+=" --test_dir ${test_dataset_path}"
OPTS+=" --deepspeed ${DS_CONFIG}"
OPTS+=" --num_train_epochs ${EPOCHS}"
OPTS+=" --logging_steps ${logging_steps}"
OPTS+=" --merge_adapters"
OPTS+=" --use_flash_attn"
OPTS+=" --output_dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --hf_token ${HF_TOKEN}"




CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/krikri_finetuning.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log