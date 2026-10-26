# 灵境 AI 方向第二阶段项目

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