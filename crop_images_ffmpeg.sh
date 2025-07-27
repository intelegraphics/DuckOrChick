#!/bin/bash

# 检查是否传入目录参数
if [ $# -eq 0 ]; then
    echo "错误：请提供图片目录路径"
    echo "用法: $0 /path/to/images"
    exit 1
fi

target_dir="$1"
old_dir="${target_dir}/old"

# 支持的图片格式（ffmpeg 支持的格式）
image_extensions=("jpg" "jpeg" "png" "bmp" "tiff")

# 检查 ffmpeg 是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo "错误：ffmpeg 未安装，请先安装"
    echo "Ubuntu/Debian: sudo apt install ffmpeg"
    echo "CentOS/RHEL: sudo yum install ffmpeg"
    echo "MacOS: brew install ffmpeg"
    exit 1
fi

# 创建 old 目录（存放原始图片）
mkdir -p "$old_dir"

# 计数器
count=0

# 遍历所有支持的图片格式
for ext in "${image_extensions[@]}"; do
    while IFS= read -r -d $'\0' image; do
        # 获取文件名（不含路径）
        filename=$(basename "$image")

        # 原始图片移动到 old 目录
        mv "$image" "$old_dir/$filename"

        # 裁剪后的图片保存在原始目录（覆盖式）
        ffmpeg -i "$old_dir/$filename" -vf "scale=224:224:force_original_aspect_ratio=increase,crop=224:224" -loglevel error "$target_dir/$filename"

        ((count++))
        echo "已处理: $filename (原始文件已备份到 $old_dir)"
    done < <(find "$target_dir" -maxdepth 1 -type f -iname "*.${ext}" -print0)
done

echo "完成！共处理 $count 张图片"
echo "原始文件备份至: $old_dir"
echo "裁剪后的图片保存在: $target_dir"