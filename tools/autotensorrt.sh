#!/bin/sh

set -euo pipefail
# config="$PWD/tools/config"
[[ "$#" == 0 ]] && echo "please give a config file..." && exit 1
config="$1"

[[ (-f "$config") ]] || (echo "\"$config\" does not exist." && exit 1)
[[ (-f "$PWD/gen_wts.py") ]] || (echo "$PWD/gen_wts.py does not exist." && exit 1)

echo "Reading config from \"$config\"..."
width="$(awk -F '=' '/^width/ {print $2}' "$config")"
height="$(awk -F '=' '/^height/ {print $2}' "$config")"
class="$(awk -F '=' '/^class/ {print $2}' "$config")"
pt="$(awk -F '=' '/^pt_path/ {print $2}' "$config")"
type="$(awk -F '=' '/^model_type/ {print $2}' "$config")"
engine="$(awk -F '=' '/^engine_path/ {print $2}' "$config")"
build="$(awk -F '=' '/^build_dir/ {print $2}' "$config")"
tensorrtx="$(awk -F '=' '/^tensorrtx_dir/ {print $2}' "$config")"
build="$tensorrtx/$build"

# check str
[[ -n "$type" ]] || (echo "Please check $config, there is no \"model_type\"..." && exit 1)
[[ -n "$width" ]] || (echo "Please check $config, there is no \"width\"..." && exit 1)
[[ -n "$height" ]] || (echo "Please check $config, there is no \"height\"..." && exit 1)
[[ -n "$class" ]] || (echo "Please check $config, there is no \"class\"..." && exit 1)
[[ -n "$pt" ]] || (echo "Please check $config, there is no \"pt_path\"..." && exit 1)
[[ -n "$engine" ]] || (echo "Please check $config, there is no \"engine_path\"..." && exit 1)
[[ -n "$tensorrtx" ]] || (echo "Please check $config, there is no \"tensorrtx_dir\"..." && exit 1)
[[ -n "$build" ]] || (echo "Please check $config, there is no \"build_dir\"..." && exit 1)

# check exist
[[ -d "$tensorrtx" ]] || (echo "$tensorrtx does not exist." && exit 1)
[[ -f "$pt" ]] || (echo "$pt does not exist." && exit 1)
[[ -f "$tensorrtx/yololayer.h" ]] || (echo "$tensorrtx/yololayer.h does not exist." && exit 1)
[[ -d "$(dirname "$engine")" ]] || mkdir -p "$(dirname "$engine")"

echo "model type: $type"
echo "width: $width"
echo "height: $height"
echo "class number: $class"
echo "weight path: \"$pt\""
echo "engine will save in \"$engine\""
echo "tensorrtx file is \"$tensorrtx\""
echo "build file is \"$build\""
echo '-------------------------------------------'

sed -i "/^\s*static constexpr int INPUT_W/s/[0-9]\+/$width/" "$tensorrtx/yololayer.h"
sed -i "/^\s*static constexpr int INPUT_H/s/[0-9]\+/$height/" "$tensorrtx/yololayer.h"
sed -i "/^\s*static constexpr int CLASS_NUM/s/[0-9]\+/$class/" "$tensorrtx/yololayer.h"

if [[ -d "$build" ]]; then
	echo "\"$build\" existed, Clear it?"
	read -p "[y]es or [n]o (default: no) : " -r option1
	[[ "$option1" == 'y' || "$option1" == 'yes' ]] && rm -rf "$build"
	echo '-------------------------------------------'
fi
mkdir -p "$build"

echo "Generating the wts file..."
python3 gen_wts.py -w "$pt" -o "$build/temp.wts"
echo "finished"
echo '-------------------------------------------'

echo "Compiling the cpp code..."
cd "$build"
cmake .. &>/dev/null && make -j12 &>/dev/null
echo "finished"
echo '-------------------------------------------'

echo "Building the engine to \"$engine\"..."
echo "please wait for a while..."
./yolov5 -s "$PWD/temp.wts" "$engine" "$type" &>/dev/null
# ./yolov5 -s "$PWD/temp.wts" "$engine" "$type"
echo "finished"
echo '-------------------------------------------'

echo "Delete the temp.wts?"
read -p "[y]es or [n]o (default: no) : " -r option2
[[ "$option2" == 'y' || "$option2" == 'yes' ]] && rm "$build/temp.wts"
echo "finished"
echo '-------------------------------------------'
cd - &>/dev/null
