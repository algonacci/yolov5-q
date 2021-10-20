#!/bin/sh

set -euo pipefail
# assume the `data_dir` like below
# ├── data_dir
# │   ├── pic1.jpg
# │   ├── pic1.xml
# │   ├── pic2.jpg
# │   ├── pic2.xml
# │   ├── .
# │   ├── .
# │   ├── .

# assume the `target_dir` like below or `empty`
# ├── target_dir
# │   ├── images
# │   │   ├── train
# │   │   ├── val
# │   ├── labels
# │   │   ├── train
# │   │   ├── val

# if target_dir="", `data_dir` will be chosen

data_dir="/e/datasets/phone_all/phone0624_/ori"
target_dir="/e/datasets/phone_all/phone0624_"
script="/d/projects/data_tools/old_code/voc2yolo.py"
split="/d/projects/data_tools/split_data.py"
split_rate=0.8
class_names=("phone" "play_phone" "sleep")
new_yaml="./data/test.yaml"

echo "Make sure that there are only \`pictures\` and \`xml file\` \
 with the same names corresponding each other."
echo "like below?"
cat <<_EOF_
├── "$data_dir"
│   ├── pic1.jpg
│   ├── pic1.xml
│   ├── pic2.jpg
│   ├── pic2.xml
│   ├── .
│   ├── .
│   ├── .
_EOF_
read -p "[y]es or [n]o (default: no) : " -r option1
echo "---------------------------------"
[[ "$option1" == 'n' || "$option1" == 'no' ]] && exit 1

if [[ -n "$target_dir" ]]; then
	echo "Make sure you \`target_dir\` empty or "
	echo "like below?"
	cat <<_EOF_
├── "$target_dir"
│   ├── images
│   │   ├── train
│   │   ├── val
│   ├── labels
│   │   ├── train
│   │   ├── val
_EOF_
	read -p "[y]es or [n]o (default: no) : " -r option2
	echo "---------------------------------"
	[[ "$option2" == 'n' || "$option2" == 'no' ]] && exit 1
fi

img_dir="$data_dir/images"
xml_dir="$data_dir/xmls"

mkdir -p "$xml_dir" && mv -i $data_dir/*.xml "$xml_dir" #  && mv -i "$xml_dir" "$target_dir"
mkdir -p "$img_dir" && mv -i $data_dir/*.* "$img_dir"   # && mv -i "$img_dir" "$target_dir"

echo "Xml to txt..."
python3 "$script" --data-dir "$data_dir" --class-names "${class_names[@]}" # &>/dev/null
echo "finished"
echo "---------------------------------"

echo "Spliting data with split rate $split_rate.."
python3 "$split" --data-dir "$data_dir" --split-rate $split_rate # &>/dev/null
echo "finished"
echo "---------------------------------"

if [[ -n "$target_dir" ]]; then
	mkdir -p "$target_dir"
	# if [[ -z "$(ls "$target_dir")" ]]; then
	if [[ !(-d "$target_dir/images") && !(-d "$target_dir/labels") ]]; then
		# mv -iu "$xml_dir" "$target_dir"
		mv -iu "$img_dir" "$target_dir"
		mv -iu "$data_dir/labels" "$target_dir"
	else
		mv -iu $img_dir/train/* "$target_dir/images/train"
		mv -iu $img_dir/val/* "$target_dir/images/val"
		mv -iu $data_dir/labels/train/* "$target_dir/labels/train/"
		mv -iu $data_dir/labels/val/* "$target_dir/labels/val/"
	fi
	data_dir="$target_dir"
fi

# generte data.yaml
echo "Generating data.yaml..."
[ -f "$new_yaml" ] && rm "$new_yaml"
echo "path: $data_dir" >>$new_yaml
echo "train: images/train" >>$new_yaml
echo "val: images/val" >>$new_yaml
# echo "test: test" >>$new_yaml
echo "nc: ${#class_names[@]}" >>$new_yaml
echo "names: [${class_names[@]}]" >>$new_yaml
sed -i '/^names/s/ /, /2g' "$new_yaml"
echo "finished"
echo "---------------------------------"

# python ./train.py --data "$new_yaml" --weights weights/yolov5s.pt --img 640

# split_rate=8/10
# let train_rate=$(echo $split_rate | cut -d'/' -f1)
# let all_rate=$(echo $split_rate | cut -d'/' -f2)
# (($((train_rate / all_rate)) > 0)) && echo "\`split_rate\` should be less than 1, please checkout!" && exit 1

# for i in train val; do
# 	mkdir -p "$data_dir/images/$i"
# done
# let total=$(ls "$data_dir/images" | wc -l)
# let train_num=$((total / all_rate * train_rate))
# echo $train_num
# mv $(find "$data_dir/images" -type f | shuf -n $train_num) "$data_dir/images/train"
# mv $(find "$data_dir/images" -maxdepth 1 -type f) "$data_dir/images/val"

# [[ ! (-d "$data_dir/images") ]] && echo "$data_dir/images does not exist, please checkout!" && exit 1
# [[ ! (-d "$data_dir/xmls") ]] && echo "$data_dir/xmls does not exist, please checkout!" && exit 1

# if [[ -d "$data_dir/labels" ]]; then
# 	echo "\"$data_dir/labels\" existed, Remove it?"
# 	read -p "[y]es or [n]o (default: no) : " -r option1
# 	[[ "$option1" == 'y' || "$option1" == 'yes' ]] &&
# 		rm -rf "$data_dir/labels" && echo "\"$data_dir/labels\" Removed."
# 	echo '-------------------------------------------'
# fi
