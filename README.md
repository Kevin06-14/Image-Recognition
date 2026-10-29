# 第二阶段项目

## 当前功能

- YOLO：检测图片中的物体、类别、置信度和边界框。
- YOLO + Mobile-SAM：根据检测框生成像素级分割结果。
- 视觉大模型问答：对同一张图片进行描述，并保存回答结果。

## 运行视觉大模型问答

首次使用时复制配置模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填写视觉模型服务商提供的 `LLM_API_KEY`。然后运行：

```powershell
python src\vision_qa.py
```

结果保存在 `outputs/vision_qa/`。`.env` 只在本机使用，不要提交到 Git。

## 生成两种方案的对比报告

在 YOLO、SAM 和视觉大模型都运行过后，执行：

```powershell
python src\compare_results.py
```

报告保存在 `outputs/comparison_report.md`。

## 一键运行第二阶段核心流程

确认 `data/test.jpg` 已准备好、`mobile_sam.pt` 已下载后，可以执行：

```powershell
python src\run_stage2.py
```

程序会依次运行 YOLO 检测、Mobile-SAM 分割、视觉大模型问答和结果对比报告。
视觉大模型需要 `.env` 中的 API 配置；如果网络或额度暂时不可用，本地检测和分割结果仍会保留。

## 第三阶段：本地 Web 服务（已开始）

第三阶段先把第二阶段的两套视觉方案封装成统一服务：

- `POST /api/analyze`：上传一张图片，同时运行 YOLO、Mobile-SAM 和视觉大模型。
- `GET /docs`：查看 FastAPI 自动生成的接口文档。
- `GET /`：打开最简单的网页入口。

启动方式：

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

然后浏览器打开 `http://127.0.0.1:8000/`，上传图片即可。首次分析仍需要项目根目录中的
`yolo11n.pt`、`mobile_sam.pt`，视觉大模型功能还需要本机 `.env` 配置 API。

当前已经完成本地 Web 化、SQLite 历史记录和基础登录鉴权，后续将继续增加 Docker 部署和服务器运行配置。

### 分析历史记录

网页每次分析成功后，会自动写入项目 `data/db/analysis.sqlite3`。该数据库目录已加入
`.gitignore`，不会提交图片、数据库或个人数据。最近记录可以通过以下接口查看：

```text
http://127.0.0.1:8000/api/history
```

### 登录鉴权

网页现在支持注册和登录。首次使用时输入用户名和至少 6 位密码，点击“注册”，
然后点击“登录”。登录成功后，浏览器会保存 JWT 令牌，并自动把令牌带到图片分析、
历史记录和结果图片请求中。

对应接口：

```text
POST /api/register
POST /api/login
```

建议在 `.env` 中设置随机的 `AUTH_SECRET`。不要把 `.env` 或真实密码提交到 Git。

### Docker 部署

项目提供了 `Dockerfile` 和 `docker-compose.yml`。由于模型权重较大且不应提交到 Git，
启动容器前请确认项目根目录存在：

```text
yolo11n.pt
mobile_sam.pt
```

首次使用时准备环境文件并启动：

```powershell
Copy-Item .env.example .env
docker compose up --build
```

浏览器访问 `http://localhost:8000/`。停止服务：

```powershell
docker compose down
```

上传图片、SQLite 数据库和分析结果通过目录挂载保存在宿主机的 `data/` 和 `outputs/`
中；容器删除后数据仍然保留。Windows 本地没有 Docker 时，可在 Linux 服务器上执行相同命令。

