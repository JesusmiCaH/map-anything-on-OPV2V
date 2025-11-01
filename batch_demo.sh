trainset_path="/data/train"

# get every folder in trainset_path
for folder in $(ls $trainset_path); do
    echo "Processing Inference: $folder"
    python demo_multi_model.py --datapath "$trainset_path" --current_time "$folder"
    done
