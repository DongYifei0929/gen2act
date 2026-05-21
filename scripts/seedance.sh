#!/bin/bash

# 检查是否为查询模式
if [[ "$1" == "--query" ]]; then
    # 可选：指定下载文件夹，默认为 ../downloaded_videos
    DOWNLOAD_DIR="${2:-../downloaded_videos}"
    
    AUTH_TOKEN="ark-0e8c64d4-c0aa-40e7-a3f4-6e2bd929add9-b691a"
    API_URL="https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    
    # 读取任务ID
    if [[ ! -f "task_ids.txt" ]]; then
        echo "错误: task_ids.txt 不存在，请先运行脚本生成任务"
        exit 1
    fi
    
    # 创建下载文件夹
    mkdir -p "$DOWNLOAD_DIR"
    echo "下载文件夹: $DOWNLOAD_DIR"
    echo ""
    echo "查询任务状态并下载完成的视频..."
    echo ""
    
    # 逐行读取任务ID并查询
    count=1
    while IFS= read -r TASK_ID; do
        if [[ -z "$TASK_ID" ]]; then
            continue
        fi
        
        echo "查询任务 $count: $TASK_ID"
        
        # 使用curl发送GET请求查询任务状态（对照官方脚本）
        response=$(curl -s -X GET "${API_URL}/${TASK_ID}" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${AUTH_TOKEN}")
        
        echo "状态响应:"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        
        # 解析响应并下载视频
        export DOWNLOAD_DIR="$DOWNLOAD_DIR"
        python3 << 'PYTHON_EOF'
import json
import sys
import os

response_str = sys.stdin.read()

try:
    response_json = json.loads(response_str)
    
    # 检查任务状态
    status = response_json.get('status', '')
    task_id = response_json.get('id', '')
    
    print(f"\n任务状态: {status}")
    
    # 如果任务完成，提取视频URL并下载
    if status == 'succeeded':
        content_info = response_json.get('content', {})
        if isinstance(content_info, dict):
            print(f"Content信息: {content_info}")
            video_url = content_info.get('video_url', '')
            if video_url:
                print(f"视频URL: {video_url}")
                print(f"开始下载视频...")
                
                # 使用curl下载视频
                download_dir = os.environ.get('DOWNLOAD_DIR', './downloaded_videos')
                os.makedirs(download_dir, exist_ok=True)
                filename = os.path.join(download_dir, f"{task_id}.mp4")
                download_cmd = f'curl -s -L "{video_url}" -o "{filename}"'
                exit_code = os.system(download_cmd)
                
                if exit_code == 0:
                    print(f"✓ 视频已保存到: {filename}")
                else:
                    print(f"✗ 下载失败: {filename}")
            else:
                print("未找到视频URL")
        else:
            print("Content 格式不正确")
    else:
        print(f"任务未完成，当前状态: {status}")
        
except json.JSONDecodeError:
    print("无法解析JSON响应")
except Exception as e:
    print(f"处理响应出错: {e}")
PYTHON_EOF < <(echo "$response")
        
        echo ""
        ((count++))
    done < task_ids.txt
    exit 0
fi

# ===== 正常发送请求模式 =====

python3 << 'PYTHON_EOF'
import json
import urllib.request
import sys
import time


def read_images_from_log(path):
    encodings = ['utf-8', 'utf-16-le', 'utf-16']

    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                lines = [line.strip().lstrip('\ufeff') for line in f.readlines() if line.strip()]
            return lines
        except:
            continue

    raise RuntimeError(f'无法读取 {path}')

# 1. 读取prompts.txt
with open('prompts.txt', 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# 2. 读取base_64.log (处理UTF-16编码)
images = read_images_from_log('base_64.log')

# 检查数据
print(f"读取到 {len(prompts)} 个提示和 {len(images)} 个图片")

if len(prompts) != len(images):
    print(f"错误: prompts 和 images 的数量不匹配，prompts: {len(prompts)}, images: {len(images)}")
    sys.exit(1)

# API 配置
API_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
AUTH_TOKEN = "ark-0e8c64d4-c0aa-40e7-a3f4-6e2bd929add9-b691a"

# 保存所有任务ID
task_ids = []

# 循环发送三个请求
for i in range(len(prompts)):
    print(f"\n发送请求 {i+1}/{len(prompts)}...")
    
    text_content = prompts[i]
    image_url = images[i]
    
    print(f"文本: {text_content[:50]}...")
    print(f"图片URL长度: {len(image_url)}")
    
    # 构建请求体
    request_body = {
        "model": "doubao-seedance-2-0-260128",
        "content": [
            {
                "type": "text",
                "text": text_content
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
                "role": "first_frame"
            }
        ],
        "generate_audio": False,
        "ratio": "16:9",
        "duration": 5,
        "watermark": False
    }
    
    # 转换为JSON字符串
    json_data = json.dumps(request_body, ensure_ascii=False).encode('utf-8')
    
    # 创建请求
    req = urllib.request.Request(
        API_URL,
        data=json_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AUTH_TOKEN}"
        }
    )
    
    # 发送请求
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = response.read().decode('utf-8')
            result_json = json.loads(result)
            task_id = result_json.get('id')
            if task_id:
                task_ids.append(task_id)
            print(f"状态码: {response.status}")
            print(f"任务ID: {task_id}")
    except urllib.error.HTTPError as e:
        error_response = e.read().decode('utf-8')
        print(f"状态码: {e.code}")
        print(f"错误响应: {error_response}")
    except Exception as e:
        print(f"请求失败: {e}")
    
    print("---")

# 保存任务ID到文件
if task_ids:
    with open('task_ids.txt', 'w') as f:
        for task_id in task_ids:
            f.write(task_id + '\n')
    print(f"\n已保存 {len(task_ids)} 个任务ID到 task_ids.txt")
    print("\n任务ID列表:")
    for i, task_id in enumerate(task_ids, 1):
        print(f"  {i}. {task_id}")

print("\n所有请求发送完成")
print("\n提示: 您可以用这些ID查询生成进度。使用以下命令查询结果:")
print("  bash seedance.sh --query")
PYTHON_EOF