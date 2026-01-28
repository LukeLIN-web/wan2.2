#!/usr/bin/env python3
"""
视频播放器Web应用
左边显示视频描述，右边显示视频
在 127.0.0.1:8080 访问
"""

import http.server
import socketserver
import os
import json
import urllib.parse
import re
from pathlib import Path

PORT = 8080
# 视频目录
VIDEO_DIRS = [
    # "/home/user1/workspace/juyi/Wan2.2",
    "/home/user1/workspace/juyi/Wan2.2/generatedvideo",
    "/home/user1/workspace/juyi/Wan2.2/v2v"
]

def get_video_files():
    """获取所有视频文件及其描述，按文件夹分组"""
    folders = []
    for video_dir in VIDEO_DIRS:
        if not os.path.exists(video_dir):
            continue
        folder_name = os.path.basename(video_dir) or video_dir
        videos = []
        for filename in os.listdir(video_dir):
            if filename.endswith(('.mp4', '.webm', '.mov', '.avi')):
                filepath = os.path.join(video_dir, filename)
                # 从文件名提取描述
                description = extract_description(filename)
                videos.append({
                    'filename': filename,
                    'path': filepath,
                    'description': description,
                    'size': format_size(os.path.getsize(filepath)),
                    'folder': folder_name
                })
        if videos:
            folders.append({
                'folder': folder_name,
                'path': video_dir,
                'videos': videos
            })
    return folders

def extract_description(filename):
    """从文件名中提取描述文字"""
    # 去掉扩展名
    name = os.path.splitext(filename)[0]
    
    # 匹配格式: xxx_1280*720_1_描述_日期时间.mp4
    match = re.search(r'\d+\*\d+_\d+_(.+?)_\d{8}_\d{6}$', name)
    if match:
        desc = match.group(1)
        # 将下划线替换为空格
        desc = desc.replace('_', ' ')
        return desc
    
    # 如果没有匹配到特定格式，返回清理后的文件名
    return name.replace('_', ' ')

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频播放器</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e4e4e4;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        /* 左侧视频列表 */
        .sidebar {
            width: 350px;
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
            padding: 20px;
        }
        
        .sidebar h1 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #00d9ff;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }
        
        .video-list {
            list-style: none;
        }
        
        .folder-group {
            margin-bottom: 20px;
        }
        
        .folder-header {
            font-size: 0.9rem;
            font-weight: 600;
            color: #ff9f43;
            padding: 10px 15px;
            background: rgba(255, 159, 67, 0.1);
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #ff9f43;
        }
        
        .video-item {
            padding: 15px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        
        .video-item:hover {
            background: rgba(0, 217, 255, 0.1);
            border-color: rgba(0, 217, 255, 0.3);
            transform: translateX(5px);
        }
        
        .video-item.active {
            background: rgba(0, 217, 255, 0.15);
            border-color: #00d9ff;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        }
        
        .video-item .title {
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 5px;
            line-height: 1.4;
            color: #fff;
        }
        
        .video-item .meta {
            font-size: 0.75rem;
            color: #888;
        }
        
        /* 右侧视频播放区域 */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 30px;
        }
        
        .video-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 20px;
            overflow: hidden;
            position: relative;
        }
        
        .video-container video {
            max-width: 100%;
            max-height: 100%;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }
        
        .video-placeholder {
            text-align: center;
            color: #666;
        }
        
        .video-placeholder .icon {
            font-size: 4rem;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        /* 视频描述区域 */
        .description-panel {
            margin-top: 20px;
            padding: 20px 25px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .description-panel h2 {
            font-size: 1rem;
            color: #00d9ff;
            margin-bottom: 10px;
        }
        
        .description-panel .filename {
            font-size: 0.85rem;
            line-height: 1.4;
            color: #888;
            word-break: break-all;
            margin-bottom: 10px;
            font-family: 'Consolas', 'Monaco', monospace;
            background: rgba(0, 0, 0, 0.2);
            padding: 8px 12px;
            border-radius: 6px;
        }
        
        .description-panel .desc-text {
            font-size: 1rem;
            line-height: 1.6;
            color: #ccc;
        }
        
        /* 滚动条样式 */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 217, 255, 0.3);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 217, 255, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>📹 视频列表</h1>
            <ul class="video-list" id="videoList"></ul>
        </div>
        <div class="main-content">
            <div class="video-container" id="videoContainer">
                <div class="video-placeholder">
                    <div class="icon">🎬</div>
                    <p>请从左侧选择一个视频播放</p>
                </div>
            </div>
            <div class="description-panel" id="descriptionPanel">
                <h2>📁 文件信息</h2>
                <div class="filename" id="videoFilename">选择视频后显示文件名</div>
                <div class="desc-text" id="videoDescription">选择视频后显示描述信息</div>
            </div>
        </div>
    </div>

    <script>
        let folders = [];
        let allVideos = [];
        let currentIndex = -1;
        
        // 加载视频列表
        fetch('/api/videos')
            .then(res => res.json())
            .then(data => {
                folders = data;
                // 展平所有视频用于索引
                allVideos = folders.flatMap(f => f.videos);
                renderVideoList();
            });
        
        function renderVideoList() {
            const list = document.getElementById('videoList');
            let html = '';
            let globalIndex = 0;
            
            folders.forEach(folder => {
                html += `<div class="folder-group">`;
                html += `<div class="folder-header">📂 ${folder.folder}</div>`;
                folder.videos.forEach(video => {
                    html += `
                        <li class="video-item" onclick="playVideo(${globalIndex})" data-index="${globalIndex}">
                            <div class="title">${video.description}</div>
                            <div class="meta">${video.size}</div>
                        </li>
                    `;
                    globalIndex++;
                });
                html += `</div>`;
            });
            
            list.innerHTML = html;
        }
        
        function playVideo(index) {
            currentIndex = index;
            const video = allVideos[index];
            
            // 更新活动状态
            document.querySelectorAll('.video-item').forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });
            
            // 播放视频
            const container = document.getElementById('videoContainer');
            container.innerHTML = `
                <video controls autoplay>
                    <source src="/video/${encodeURIComponent(video.filename)}?path=${encodeURIComponent(video.path)}" type="video/mp4">
                    您的浏览器不支持视频播放
                </video>
            `;
            
            // 更新文件名和描述
            document.getElementById('videoFilename').textContent = `[📂 ${video.folder}] ${video.filename}`;
            document.getElementById('videoDescription').textContent = video.description;
        }
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp' && currentIndex > 0) {
                playVideo(currentIndex - 1);
            } else if (e.key === 'ArrowDown' && currentIndex < allVideos.length - 1) {
                playVideo(currentIndex + 1);
            }
        });
    </script>
</body>
</html>
'''

class VideoHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif path == '/api/videos':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            videos = get_video_files()
            self.wfile.write(json.dumps(videos, ensure_ascii=False).encode('utf-8'))
            
        elif path.startswith('/video/'):
            # 获取视频路径
            query = urllib.parse.parse_qs(parsed_path.query)
            video_path = query.get('path', [''])[0]
            
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                
                # 支持 Range 请求（用于视频 seeking 和流式传输）
                range_header = self.headers.get('Range')
                
                if range_header:
                    range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
                    if range_match:
                        start = int(range_match.group(1))
                        end = int(range_match.group(2)) if range_match.group(2) else min(start + 1024*1024, file_size - 1)
                        
                        # 限制每次传输块大小为1MB，避免卡顿
                        chunk_size = end - start + 1
                        
                        self.send_response(206)
                        self.send_header('Content-Type', 'video/mp4')
                        self.send_header('Accept-Ranges', 'bytes')
                        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                        self.send_header('Content-Length', str(chunk_size))
                        self.end_headers()
                        
                        with open(video_path, 'rb') as f:
                            f.seek(start)
                            # 分块读取和写入
                            remaining = chunk_size
                            while remaining > 0:
                                read_size = min(65536, remaining)  # 64KB chunks
                                data = f.read(read_size)
                                if not data:
                                    break
                                self.wfile.write(data)
                                remaining -= len(data)
                        return
                
                # 没有Range请求时，返回部分内容让浏览器发起Range请求
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(file_size))
                self.end_headers()
                
                # 流式传输，每次64KB
                with open(video_path, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            break
            else:
                self.send_error(404, 'Video not found')
        else:
            self.send_error(404, 'Not found')
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

def kill_port(port):
    """杀掉占用指定端口的进程"""
    import subprocess
    import signal
    
    # 尝试多种方法
    methods = [
        f"fuser -k {port}/tcp",
        f"lsof -ti:{port} | xargs -r kill -9",
    ]
    
    for cmd in methods:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"🔄 已杀掉占用端口 {port} 的进程")
                import time
                time.sleep(1)
                return True
        except Exception:
            continue
    return False

# 允许端口复用的TCP服务器
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def main():
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with ReusableTCPServer(("127.0.0.1", PORT), VideoHandler) as httpd:
                print(f"🎬 视频播放器已启动!")
                print(f"📺 请在浏览器访问: http://127.0.0.1:{PORT}")
                print(f"📁 视频目录: {VIDEO_DIRS}")
                print(f"⏹  按 Ctrl+C 停止服务器")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n👋 服务器已停止")
                break
        except OSError as e:
            if "Address already in use" in str(e) and attempt < max_retries - 1:
                print(f"⚠️  端口 {PORT} 被占用，尝试清理... (尝试 {attempt + 1}/{max_retries})")
                kill_port(PORT)
                time.sleep(1)
                continue
            else:
                raise

if __name__ == "__main__":
    main()

