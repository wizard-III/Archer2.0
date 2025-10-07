#!/bin/bash

export WANDB_API_KEY=your_wandb_api_key
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
LOG_FILE="lcb_evaluated_tasks.log"

# 一级目录筛选配置, 使用空格分隔的目录名列表，如 "DAPO-Qwen2.5-1.5B-Math-01 DAPO-Qwen2.5-1.5B-Math-02"
# 留空则检测所有一级目录
TARGET_DIRS=${TARGET_DIRS:-"DAPO-Qwen2.5-1.5B-Math-01 DAPO-Qwen2.5-1.5B-Code-01"}  # 通过环境变量设置

# 全局变量存储监控进程信息
declare -A MONITOR_PIDS

touch "$LOG_FILE"

# 函数：记录任务完成状态
log_task_completion() {
    local model_path="$1"
    echo "$(date +%s)|$model_path" >> "$LOG_FILE"
}

# 函数：执行单个评测任务
run_evaluation() {
    local model_path="$1"
    
    echo "[$(date)] 开始评测: $model_path"

    project_name=ArcherEval
    experiment_name=$(basename "$(dirname "$model_path")")
    global_step=$(basename "$model_path" | awk -F'_' '{print $3}')

    python LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics_online.py \
        --eval_file ${model_path}/actor/hf_model/output/livecodebench_v5.parquet \
        --project_name ${project_name} \
        --experiment_name ${experiment_name} \
        --global_step ${global_step}

    # 检查评测是否成功
    local result_file="$model_path/actor/hf_model/output/livecodebench_v5.parquet.pass.lcb.csv"
    if [ $? -eq 0 ] && [ -f "$result_file" ]; then
        log_task_completion "$model_path"
        echo "[$(date)] 评测成功: $model_path"
        python tools/sync_offline_eval_to_wandb.py \
            --project_name ${project_name} \
            --experiment_name ${experiment_name}
    else
        echo "[$(date)] 错误：评测失败或结果文件未生成: $model_path"
        return 1
    fi
}

is_infer_completed() {
    local model_path="$1"
    [ -f "$model_path/actor/hf_model/output/livecodebench_v5.parquet" ]
}

is_eval_completed() {
    local model_path="$1"
    [ -f "$model_path/actor/hf_model/output/livecodebench_v5.parquet.pass.lcb.csv" ]
}

process_model() {
    local model_path="$1"
    if is_infer_completed "$model_path" && ! is_eval_completed "$model_path" ; then
        echo "[$(date)] 发现新任务: $model_path"
        run_evaluation "$model_path"
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
