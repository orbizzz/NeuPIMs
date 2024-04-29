#!/bin/bash
cd build; make -j; cd ..;

# config file
config="./configs/systolic_ws_128x128_dev.json"
mem_config_dir="./neupims_configs/memory_configs"
cli_config_dir="./neupims_configs/client_configs"
model_config_dir="./neupims_configs/model_configs"
sys_config_dir="./neupims_configs/system_configs"

program="./build/bin/Simulator"

# log
LOG_LEVEL=info
DATE=$(date "+%F_%H:%M:%S")
LOG_ROOT="experiment_logs/${DATE}"

i=0

for mem_config in "$mem_config_dir"/*
do
    for cli_config in "$cli_config_dir"/*
    do
        for model_config in "$model_config_dir"/*
        do
            for sys_config in "$sys_config_dir"/*
            do
                EXP_ID=${sys_config//$sys_config_dir/""}
                EXP_ID=${EXP_ID//".json"/""}
                EXP_ID=exp#$i-${EXP_ID//\/}
                LOG_DIR="${LOG_ROOT}/$EXP_ID"

                mkdir -p $LOG_DIR
                LOG_NAME=simulator.log
                CONFIG_FILE=${LOG_DIR}/config.log

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
                
                i=$((i+1))
            done
        done
    done
done

