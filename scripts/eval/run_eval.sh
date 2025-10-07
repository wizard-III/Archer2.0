#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

nnodes=1
tp_size=1

project_name=Eval

# Path
base_dir=.
model_path=${base_dir}/model/Archer2.0-Code-1.5B-Preview
output_dir=${model_path}/output
data_dir=${base_dir}/data/test
dataset=livecodebench_v5 # livecodebench_v6

##### livecodebench_v5
n_samples=8
temperature=0.8
top_p=0.95
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))

# ##### livecodebench_v6
# n_samples=16
# temperature=0.8
# top_p=0.95
# max_prompt_length=$((1024 * 2))
# max_response_length=$((1024 * 32))

python -m verl.trainer.main_generation \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=8 \
    +trainer.project_name=${project_name} \
    +trainer.experiment_name=base \
    +trainer.task_name=${dataset} \
    +trainer.global_step=0 \
    +trainer.use_wandb=False \
    model.path=${model_path} \
    data.path=${data_dir}/${dataset}.parquet \
    data.output_path=${output_dir}/${dataset}.parquet \
    data.batch_size=2048 \
    data.n_samples=${n_samples} \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.tensor_model_parallel_size=${tp_size} \
    rollout.temperature=$temperature \
    rollout.top_k=-1 \
    rollout.top_p=$top_p \
    rollout.prompt_length=$max_prompt_length \
    rollout.response_length=$max_response_length \
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
