#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

# 自动化模型评测脚本 - 按任务粒度检测 + 模型转换
# 监控路径: /home/wangjiakang/data/output/Archer2.0

# ├── DAPO-Qwen2.5-1.5B-Math-01/
# │   ├── global_step_100/
# │   │   ├── actor/
# │   │   │   ├── config.json
# │   │   │   ├── ... # 其他模型文件
# │   │   │   └── hf_model/  # 转换后生成
# │   │   │       ├── config.json
# │   │   │       └── output/  # 评测结果
# │   │   │           ├── aime2024.parquet.results.json
# │   │   │           ├── aime2025.parquet.results.json
# │   │   │           └── livecodebench_v5.parquet.results.json
# │   ├── global_step_200/
# │   │   └── ... # 类似结构
# ├── DAPO-Qwen2.5-1.5B-Math-02/
# │   └── ... # 类似结构
# └── ... # 其他模型

MODEL_ROOT="/home/wangjiakang/data/output/Archer2.0"
LOG_FILE="evaluated_tasks.log"
CONVERSION_LOG="model_conversion.log"

# 一级目录筛选配置, 使用空格分隔的目录名列表，如 "DAPO-Qwen2.5-1.5B-Math-01 DAPO-Qwen2.5-1.5B-Math-02"
# 留空则检测所有一级目录
TARGET_DIRS=${TARGET_DIRS:-"DAPO-Qwen2.5-1.5B-Math-01 DAPO-Qwen2.5-1.5B-Code-01"}  # 通过环境变量设置

# 任务与结果文件映射
declare -A TASK_MAP=(
    ["lcb"]="livecodebench_v5.parquet"
    ["aime24"]="aime2024.parquet"
    ["aime25"]="aime2025.parquet"
)

# 目录组与任务映射
declare -A GROUP_TASKS=(
    ["Math"]="aime24 aime25"   # Math组执行aime24和aime25
    ["Default"]="lcb"          # 其他组执行lcb
)

# 创建日志文件
touch "$LOG_FILE"
touch "$CONVERSION_LOG"

# 获取目录所属组
get_group() {
    local dir_name="$1"
    if [[ "$dir_name" == *"Math"* ]]; then
        echo "Math"
    else
        echo "Default"
    fi
}

# 函数：检查模型是否已保存
is_model_saved() {
    local model_path="$1"
    [ -f "$model_path/actor/config.json" ]
}

# 函数：检查模型是否已转为HF格式
is_model_converted() {
    local model_path="$1"
    [ -f "$model_path/actor/hf_model/config.json" ]
}

# 函数：检查特定任务是否已完成
is_task_completed() {
    local model_path="$1"
    local task="$2"
    
    # 从映射中获取结果文件名
    local result_file="${TASK_MAP[$task]}"
    
    # 检查结果文件是否存在
    [ -f "$model_path/actor/hf_model/output/$result_file" ]
}

# 函数：记录任务完成状态
log_task_completion() {
    local model_path="$1"
    local task="$2"
    echo "$(date +%s)|$model_path|$task" >> "$LOG_FILE"
}

# 函数：检查任务是否已记录
is_task_logged() {
    local model_path="$1"
    local task="$2"
    grep -q "$model_path|$task" "$LOG_FILE"
}

# 函数：转换模型为HF格式
convert_to_hf() {
    local model_path="$1"

    echo "[$(date)] 开始转换模型: $model_path" | tee -a "$CONVERSION_LOG"
    
    # 使用实际转换命令
    local start_time=$(date +%s)
    
    # 转换命令 - 使用您提供的示例
    python -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "$model_path/actor" \
        --target_dir "$model_path/actor/hf_model" 2>&1 | tee -a "$CONVERSION_LOG"
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 验证转换是否成功
    if [ $exit_code -eq 0 ] && [ -f "$model_path/actor/hf_model/config.json" ]; then
        echo "[$(date)] 转换成功 ($duration 秒): $model_path" | tee -a "$CONVERSION_LOG"
        return 0
    else
        echo "[$(date)] 错误：转换失败 (状态码: $exit_code, 耗时: $duration 秒): $model_path" | tee -a "$CONVERSION_LOG"
        
        # 清理可能的部分转换结果
        if [ -d "$model_path/actor/hf_model" ]; then
            rm -rf "$model_path/actor/hf_model"
            echo "[$(date)] 已清理部分转换结果" | tee -a "$CONVERSION_LOG"
        fi
        
        return 1
    fi
}

# 函数：执行单个评测任务
run_evaluation() {
    local model_path="$1"
    local task="$2"
    
    echo "[$(date)] 开始评测: $model_path - $task"

    if [ "$task" = "lcb" ]; then
        n_samples=8
        dataset=livecodebench_v5
        temperature=0.8
        top_p=0.95
    elif [ "$task" = "aime24" ]; then
        n_samples=32
        dataset=aime2024
        temperature=0.8
        top_p=0.95
    elif [ "$task" = "aime25" ]; then
        n_samples=32
        dataset=aime2025
        temperature=0.8
        top_p=0.95
    else
        echo "Unknown task: $task"
        return 1
    fi

    nnodes=1
    tp_size=1

    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 32))

    base_dir=/ytech_milm/lixiaoran/data/public/pre-release
    data_dir=${base_dir}/data/test
    output_dir=${model_path}/actor/hf_model/output
    
    project_name=ArcherEval
    experiment_name=$(basename "$(dirname "$model_path")")
    global_step=$(basename "$model_path" | awk -F'_' '{print $3}')

    python -m verl.trainer.main_generation \
        trainer.nnodes=${nnodes} \
        trainer.n_gpus_per_node=8 \
        +trainer.project_name=${project_name} \
        +trainer.experiment_name=${experiment_name} \
        +trainer.task_name=${dataset} \
        +trainer.global_step=${global_step} \
        model.path=${model_path}/actor/hf_model \
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

    # 检查评测是否成功
    local result_file="$model_path/actor/hf_model/output/${TASK_MAP[$task]}"
    if [ $? -eq 0 ] && [ -f "$result_file" ]; then
        # 记录任务完成
        log_task_completion "$model_path" "$task"
    else
        echo "[$(date)] 错误：评测失败或结果文件未生成: $model_path - $task"
        return 1
    fi
}

# 函数：处理单个模型路径
process_model() {
    local model_path="$1"
    local exp_dir=$(basename $(dirname "$model_path"))
    local group=$(get_group "$exp_dir")
    local tasks_to_run=${GROUP_TASKS[$group]}

    if [ -z "$tasks_to_run" ]; then
        echo "[$(date)] 警告：未找到 $exp_dir 的任务配置，跳过"
        return
    fi

    # 检查是否为有效模型路径
    if is_model_saved "$model_path"; then
        # 检查模型是否已转换为HF格式
        if ! is_model_converted "$model_path"; then
            echo "[$(date)] 模型未转换: $model_path"
            if convert_to_hf "$model_path"; then
                echo "[$(date)] 转换完成，准备评测任务"
            else
                echo "[$(date)] 跳过此模型，转换失败"
                return 1
            fi
        fi
        
        # 检查每个任务的完成状态
        for task in $tasks_to_run; do
            # 检查是否已完成或已记录
            if ! is_task_completed "$model_path" "$task" && ! is_task_logged "$model_path" "$task"; then
                echo "[$(date)] 发现新任务: $model_path - $task"
                
                # 添加重试机制
                max_retries=3
                attempt=1
                success=0
                
                while [ $attempt -le $max_retries ]; do
                    if run_evaluation "$model_path" "$task"; then
                        success=1
                        break
                    else
                        echo "[$(date)] 评测失败，尝试 $attempt/$max_retries"
                        sleep $((attempt * 60))  # 指数退避
                        ((attempt++))
                    fi
                done
                
                if [ $success -eq 0 ]; then
                    echo "[$(date)] 错误：任务失败超过最大重试次数: $model_path - $task"
                fi
            fi
        done
    fi
}

# 主循环
while true; do
    echo "[$(date)] 开始扫描新任务..."
    
    # 新增：一级目录筛选逻辑
    if [[ -n "$TARGET_DIRS" ]]; then
        echo "[$(date)] 使用筛选目录: $TARGET_DIRS"
        IFS=' ' read -ra TARGET_DIR_ARRAY <<< "$TARGET_DIRS"
        for dir in "${TARGET_DIR_ARRAY[@]}"; do
            full_path="${MODEL_ROOT}/${dir}"
            if [[ -d "$full_path" ]]; then
                echo "[$(date)] 扫描目录: $full_path"
                find "$full_path" -mindepth 1 -maxdepth 1 -type d -name "global_step_*" -print0 | while IFS= read -r -d '' model_path; do
                    process_model "$model_path"
                done
            else
                echo "[$(date)] 警告：目录不存在，跳过: $full_path"
            fi
        done
    else
        # 无筛选时扫描所有目录
        echo "[$(date)] 扫描所有目录 (无筛选)"
        find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d -name "global_step_*" -print0 | while IFS= read -r -d '' model_path; do
            process_model "$model_path"
        done
    fi
    
    # 检查间隔（秒）
    echo "[$(date)] 扫描完成，等待新任务..."
    sleep 60  # 每60s检查一次
done