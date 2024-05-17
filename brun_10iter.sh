#!/bin/bash
cd build; make -j; cd ..;

# config file
# npu config
config=./configs/systolic_ws_128x128_dev.json

# memory config
mem_config=./neupims_configs/memory_configs/dram.json

# client config directory
# cli_config_dir="./neupims_configs/client_configs/clb"
cli_config_dir="./neupims_configs/client_configs/rr"

# model config
model_config=./neupims_configs/model_configs/gpt3-7B.json

# sys config
sys_config=./neupims_configs/system_configs/npu_fusion.json


program="./build/bin/Simulator"
# slack_url="https://hooks.slack.com/services/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"

# log
LOG_LEVEL=info
DATE=$(date "+%F_%H:%M:%S")
LOG_ROOT="experiment_logs/${DATE}"

i=0


for cli_config in "$cli_config_dir"/*
do
    # Assign file name to a variable
    file_name=$(basename "$mem_config")
    mem_type=$(echo "$file_name" | cut -f 1 -d '.')
    file_name=$(basename "$cli_config")
    cli_type=$(echo "$file_name" | cut -f 1 -d '.')

    
    EXP_ID=#$i-${mem_type}-${cli_type}
    LOG_DIR="${LOG_ROOT}/$EXP_ID"


    mkdir -p $LOG_DIR
    LOG_NAME=simulator.log
    CONFIG_FILE=${LOG_DIR}/config.log

    echo "Experiment ID: $EXP_ID"
    echo "Log directory: $LOG_DIR"
    echo "Running NeuPIMS simulator with configurations: "
    echo "Memory configuration: $mem_config"
    echo "Client configuration: $cli_config"
    echo "Model configuration: $model_config"
    echo "System configuration: $sys_config"
    
    $program --config $config \
        --mem_config $mem_config \
        --cli_config $cli_config \
        --model_config $model_config \
        --sys_config $sys_config \
        --log_dir $LOG_DIR \
        --log_level $LOG_LEVEL > ${LOG_DIR}/${LOG_NAME}

    mem_config_name=${mem_config//$mem_config_dir/""}
    cli_config_name=${cli_config//$cli_config_dir/""}
    model_config_name=${model_config//$model_config_dir/""}
    sys_config_name=${sys_config//$sys_config_dir/""}

    echo "memory config: ${mem_config_name//\/}" > ${CONFIG_FILE}
    echo "client config: ${cli_config_name//\/}" >> ${CONFIG_FILE}
    echo "model config: ${model_config_name//\/}" >> ${CONFIG_FILE}
    echo "system config: ${sys_config_name//\/}" >> ${CONFIG_FILE}
    echo "Experiment #$i complete."
    echo ""
    # curl -X POST -H 'Content-Type: application/json' -d '{"text": "Experiment complete."}' ${slack_url}
    
    i=$((i+1))

done


