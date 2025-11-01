cd opv2v_dataset/
trainset_path="/data/train"

# get every folder in trainset_path
for folder in $(ls $trainset_path); do
    echo "Processing folder: $folder"
    python get_mutual_vis.py --datapath "$trainset_path/$folder"
    done

for folder in $(ls $trainset_path); do
    python data_sample.py --datapath "$trainset_path/$folder" --step 4 --frame_num 1
    done
cd ..
