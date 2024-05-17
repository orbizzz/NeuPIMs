cd build; make -j; cd ..;

# config file
# npu config
config=./configs/systolic_ws_128x128_dev.json

# memory config
mem_config=./neupims_configs/memory_configs/newton-like.json

# client config
# cli_config="./neupims_configs/client_configs/clb/share-gpt2-bs512-ms7B-tp4-clb-0.csv"
cli_config="./neupims_configs/client_configs/rr/share-gpt2-bs512-ms7B-tp4-rr-0.csv"

# model config
model_config=./neupims_configs/model_configs/gpt3-7B.json

# sys config
sys_config=./neupims_configs/system_configs/newton.json
# sys_config=./neupims_configs/system_configs/sub-batch-on.json
# sys_config=./neupims_configs/system_configs/sub-batch-off.json

# log file
LOG_LEVEL=info
DATE=$(date "+%F_%H:%M:%S")

LOG_DIR=experiment_logs/${DATE}

mkdir -p $LOG_DIR;
LOG_NAME=simulator.log
CONFIG_FILE=${LOG_DIR}/config.log

echo "log directory: $LOG_DIR"




if [ "$1" ]; then
    ./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir $LOG_DIR \

else
    ./build/bin/Simulator \
        --config $config \
        --mem_config $mem_config \
        --cli_config $cli_config \
        --model_config $model_config \
        --sys_config $sys_config \
        --log_dir $LOG_DIR \
        --log_level $LOG_LEVEL > ${LOG_DIR}/${LOG_NAME}
fi
    echo "memory config: $mem_config" > ${CONFIG_FILE}
    echo "client config: $cli_config" >> ${CONFIG_FILE}
    echo "model config: $model_config" >> ${CONFIG_FILE}
    echo "system config: $sys_config" >> ${CONFIG_FILE}
    cat ${CONFIG_FILE}
