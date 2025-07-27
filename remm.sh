#!/bin/bash

# 检查参数数量
if [ $# -lt 1 ]; then
    echo "错误：请提供文件夹路径作为参数"
    echo "用法: $0 /path/to/folder [前缀]"
    echo "示例: $0 ./images duck"
    exit 1
fi

folder_path="$1"
prefix="${2:-duck}"  # 如果未提供第二个参数，默认为"duck"
counter=1

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    echo "错误：文件夹不存在 '$folder_path'"
    exit 1
fi

# 遍历文件夹中的文件（不包括子目录）
for file in "$folder_path"/*; do
    if [ -f "$file" ]; then  # 确保是文件而不是目录
        # 获取文件扩展名
        filename=$(basename "$file")
        extension="${filename##*.}"
        
        # 如果是隐藏文件或无扩展名文件，特殊处理
        if [[ "$filename" == "$extension" ]]; then
            extension=""
        fi
        
        # 格式化计数器为两位数
        printf -v padded_counter "%02d" $counter
        
        # 构建新文件名
        if [ -z "$extension" ]; then
            new_name="${prefix}_${padded_counter}"
        else
            new_name="${prefix}_${padded_counter}.${extension}"
        fi
        
        # 重命名文件
        mv -n "$file" "$folder_path/$new_name" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "已重命名: $(basename "$file") -> $new_name"
            ((counter++))
        else
            echo "跳过: $(basename "$file") (可能文件名冲突)"
        fi
    fi
done

echo "重命名完成，共处理了 $((counter-1)) 个文件"
