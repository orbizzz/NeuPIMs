cd build; make -j; cd ..;

# config file
config=./configs/systolic_ws_128x128_dev.json
mem_config=./neupims_configs/memory_configs/dram.json
cli_config=./neupims_configs/client_configs/share-gpt2-bs512-ms7B-tp4-clb-0.csv
model_config=./neupims_configs/model_configs/gpt3-7B.json
sys_config=./neupims_configs/system_configs/npu_fusion.json


gdb --args ./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir .
