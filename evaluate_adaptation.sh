#!/bin/bash
# chmod +x evaluate_adaptation.sh

dataset="drebin" # "malradar"

# Array of feature names
features=("drebin") #   "mamadroid" "damd"

# Array of budgets
budgets=(10) # 20 30 40 100

# models are already trained and saved
gpu_id="1"
# Loop through each feature
for feature in "${features[@]}"
do
    if [ "${feature}" = "damd" ]; then
        batch_size="20"
    else
        batch_size="32"
    fi
    # Nested loop for each budget
    for budget in "${budgets[@]}"
    do
        python adapt_classifier.py --budget "$budget" --feature_name "$feature" --data_name "$dataset" --gpu_id "$gpu_id" --batch_size "$batch_size" 
    done
    python adapt_classifier.py --feature_name "$feature" --data_name "$dataset" --display
done
