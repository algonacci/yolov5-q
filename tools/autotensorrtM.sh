#!/bin/sh

set -euo pipefail
# config="$PWD/tools/config"
[[ "$#" == 0 ]] && echo "please give a config file..." && exit 1


total=$#
count=0
for i in $@; do
  echo '-------------------------------------------'
  config="$i"

  [[ (-f "$config") ]] || (echo "$config does not exist." && continue)
  [[ (-f "$PWD/gen_wts.py") ]] || (echo "$PWD/gen_wts.py does not exist." && continue)

  echo "Reading config from $config..."
  width="$(awk -F '=' '/^width/ {print $2}' "$config" )"
  height="$(awk -F '=' '/^height/ {print $2}' "$config" )"
  class="$(awk -F '=' '/^class/ {print $2}' "$config" )"
  pt="$(awk -F '=' '/^pt_path/ {print $2}' "$config" )"
  type="$(awk -F '=' '/^model_type/ {print $2}' "$config" )"
  engine="$(awk -F '=' '/^engine_path/ {print $2}' "$config" )"
  build="$(awk -F '=' '/^build_dir/ {print $2}' "$config" )"
  tensorrtx="$(awk -F '=' '/^tensorrtx_dir/ {print $2}' "$config" )"
  build="$tensorrtx/$build"

  # check str
  [[ -z "$type" ]] && echo "Please check $config, there is no \"model_type\"..." && continue
  [[ -z "$width" ]] && echo "Please check $config, there is no \"width\"..." && continue
  [[ -z "$height" ]] && echo "Please check $config, there is no \"height\"..." && continue
  [[ -z "$class" ]] && echo "Please check $config, there is no \"class\"..." && continue
  [[ -z "$pt" ]] && echo "Please check $config, there is no \"pt_path\"..." && continue
  [[ -z "$engine" ]] && echo "Please check $config, there is no \"engine_path\"..." && continue
  [[ -z "$tensorrtx" ]] && echo "Please check $config, there is no \"tensorrtx_dir\"..." && continue
  [[ -z "$build" ]] && echo "Please check $config, there is no \"build_dir\"..." && continue

  # check exist
  [[ ! (-d "$tensorrtx") ]] && echo "$tensorrtx does not exist." && continue
  [[ ! (-f "$tensorrtx/yololayer.h") ]] && echo "$tensorrtx/yololayer.h does not exist." && exit 1
  [[ -d "$(dirname "$engine")" ]] || mkdir -p "$(dirname "$engine")"
# 
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

  
  [[ -d "$build" ]] && echo "\"$build\" existed, Clearing it!" && \
    rm -rf "$build" && echo '-------------------------------------------'
  mkdir -p "$build"

  echo "Generating the wts file..."
  python3 gen_wts.py -w "$pt" -o "$build/temp.wts" &> /dev/null
  echo "finished"
  echo '-------------------------------------------'

  echo "Compiling the cpp code..."
  cd "$build"
  cmake .. &> /dev/null && make -j12 &> /dev/null
  echo "finished"
  echo '-------------------------------------------'

  echo "Building the engine, please wait for a while..."
  ./yolov5 -s "$PWD/temp.wts"  "$engine" "$type" &> /dev/null
  echo "finished"
  echo '-------------------------------------------'

  echo "Deleting the temp.wts!"
  rm "$build/temp.wts"
  cd - &> /dev/null
  count=$((count + 1))
done

echo "All building finished!($count/$total)"
