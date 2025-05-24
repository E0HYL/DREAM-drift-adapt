#!/bin/bash
# chmod +x plot_detection_curves.sh

dataset="malradar" # "drebin"

# Array of feature names
features=("drebin") #  "mamadroid" "damd" 

# Array of model names
models=("dream") # "basic"  "cade" "transcendent"

gpu_id="0"
# models are already trained and saved
# Loop through each feature
for feature in "${features[@]}"
do
    # Nested loop for each model
    for model in "${models[@]}"
    do
        python main.py --num_epochs 0 --model_name "$model" --feature "$feature" --data_name "$dataset" --gpu_id "$gpu_id"
    done
done
