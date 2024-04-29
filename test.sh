cd build; make -j32; cd ..;

# config file
config=./configs/systolic_ws_128x128_dev.json
mem_config=./neupims_configs/memory_configs/dram.json
cli_config=./neupims_configs/client_configs/i_128.json
model_config=./neupims_configs/model_configs/gpt2_md_layer_1.json
sys_config=./neupims_configs/system_configs/npu_fusion.json

# log file
LOG_LEVEL=info
DATE=$(date "+%F_%H:%M:%S")

LOG_DIR=experiment_logs/${DATE}

mkdir -p $LOG_DIR;
LOG_NAME=${LOG_LEVEL}_${DATE}.log
CONFIG_FILE=${LOG_DIR}/config.log

echo "log directory: $LOG_DIR"

./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir $LOG_DIR \
    --log_level $LOG_LEVEL > ${LOG_DIR}/${LOG_NAME}

# echo process_id=`/bin/ps -fu $USER| grep "Simulator" | grep -v "grep" | awk '{print $2}'`

echo "memory config: $mem_config" > ${CONFIG_FILE}
echo "client config: $cli_config" >> ${CONFIG_FILE}
echo "model config: $model_config" >> ${CONFIG_FILE}
echo "system config: $sys_config" >> ${CONFIG_FILE}
cat ${CONFIG_FILE}