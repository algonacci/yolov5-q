#!/bin/sh

data_dir="/e/datasets/phone_all/phone0624"
script="/d/projects/data_tools/old_code/voc2yolo.py"

class_names=("phone" "play_phone" "sleep")

# python3 "$script" --data-dir "$data_dir" --class-names "${class_names[@]}"

new_yaml="./data/test.yaml"

[[ -f "$new_yaml" ]] && rm "$new_yaml"

echo "path: $data_dir" >> $new_yaml
echo "train: train" >> $new_yaml
echo "val: val" >> $new_yaml
echo "test: test" >> $new_yaml

echo "nc: ${#class_names[@]}" >> $new_yaml

echo "names: [${class_names[@]}]" >> $new_yaml

sed -i '/^names/s/ /, /2g' "$new_yaml"
