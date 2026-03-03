ssh ubuntu@147.185.41.15
source neuralGaffer/bin/activate
cd Neural_Gaffer_exp


accelerate launch --main_process_port 25539 \
    --config_file configs/1_16fp.yaml \
    neural_gaffer_inference_polyhaven.py \
    --output_dir neural_gaffer_res256 \
    --mixed_precision fp16 \
    --resume_from_checkpoint latest \
    --polyhaven_data_root /home/ubuntu/LVSMExp/source_data_polyhaven/ \
    --save_dir ./polyhaven_relighting_results