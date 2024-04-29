cd build; make -j; cd ..;

# config file
config=./configs/systolic_ws_128x128_dev.json
mem_config=./neupims_configs/memory_configs/neupims.json
cli_config=./neupims_configs/client_configs/i512_bs32.json
model_config=./neupims_configs/model_configs/gpt3-7B.json
sys_config=./neupims_configs/system_configs/neupims_fusion.json


gdb --args ./build/bin/Simulator \
    --config $config \
    --mem_config $mem_config \
    --cli_config $cli_config \
    --model_config $model_config \
    --sys_config $sys_config \
    --log_dir .
