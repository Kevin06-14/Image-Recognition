"""FastAPI entry point for the stage 3 image analysis service."""

import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.auth import (
    authenticate_user,
    create_token,
    create_user,
    decode_token,
    init_auth_db,
)
from src.database import init_db, list_records, save_record
from src.inference_service import OUTPUT_DIR, analyze_image


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="图片识别双方案对比服务",
    version="3.0.0",
    description="使用 YOLO + Mobile-SAM 与视觉大模型分析同一张图片。",
)
init_auth_db()
security = HTTPBearer(auto_error=False)


class Account(BaseModel):
    username: str
    password: str


@app.get("/health")
def health() -> dict:
    """Lightweight endpoint for Docker and reverse-proxy health checks."""
    return {"status": "ok", "service": "image-recognition-stage3"}


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="请先登录。")
    try:
        return decode_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>图片识别双方案对比</title>
<style>
body{font-family:Arial;max-width:900px;margin:40px auto;line-height:1.6}
button{padding:8px 16px;cursor:pointer}.result{margin-top:24px}
.images{display:flex;gap:20px;flex-wrap:wrap}.images img{max-width:420px;border:1px solid #ddd}
pre{white-space:pre-wrap;background:#f6f6f6;padding:12px}
</style></head>
<body>
<h1>图片识别双方案对比</h1>
<p>上传一张图片，后端会同时运行 YOLO + Mobile-SAM 与视觉大模型。</p>
<form id="authForm">
  <input id="username" placeholder="用户名" required>
  <input id="password" type="password" placeholder="密码（至少6位）" required>
  <button type="button" id="register">注册</button>
  <button type="button" id="login">登录</button>
</form>
<p id="authStatus">请先登录后再分析图片。</p>
<form id="form">
  <input type="file" name="file" accept="image/*" required>
  <button type="submit">开始分析</button>
</form>
<div id="status"></div>
<section id="result" class="result" hidden>
  <h2>YOLO 检测结果</h2><pre id="detections"></pre>
  <h2>Mobile-SAM 分割结果</h2>
  <div class="images"><img id="overlay"><img id="mask"></div>
  <h2>视觉大模型回答</h2><pre id="vision"></pre>
</section>
<p>接口文档：<a href="/docs">/docs</a></p>
<p>历史记录接口：<a href="/api/history">/api/history</a></p>
<script>
const form=document.getElementById("form");
let token=localStorage.getItem("stage3_token")||"";
async function account(path){
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({username:document.getElementById("username").value,
      password:document.getElementById("password").value})});
  const data=await response.json();
  if(!response.ok) throw new Error(data.detail||"请求失败");
  if(data.access_token){token=data.access_token;localStorage.setItem("stage3_token",token);}
  document.getElementById("authStatus").textContent="登录成功，可以开始分析。";
}
async function loadProtectedImage(url, elementId){
  const response=await fetch(url,{headers:{"Authorization":"Bearer "+token}});
  if(!response.ok) throw new Error("分割结果图片读取失败："+response.status);
  const blob=await response.blob();
  document.getElementById(elementId).src=URL.createObjectURL(blob);
}
document.getElementById("register").onclick=()=>account("/api/register").catch(e=>
  document.getElementById("authStatus").textContent="错误："+e.message);
document.getElementById("login").onclick=()=>account("/api/login").catch(e=>
  document.getElementById("authStatus").textContent="错误："+e.message);
form.addEventListener("submit",async (event)=>{
  event.preventDefault();
  const status=document.getElementById("status");
  status.textContent="正在分析，请稍候（首次运行加载模型会较慢）...";
  document.getElementById("result").hidden=true;
  try{
    const response=await fetch("/api/analyze",{method:"POST",
      headers:{"Authorization":"Bearer "+token},body:new FormData(form)});
    const data=await response.json();
    if(!response.ok) throw new Error(data.detail||"分析失败");
    document.getElementById("detections").textContent=
      JSON.stringify(data.detections,null,2);
    await Promise.all([
      loadProtectedImage(data.overlay_url,"overlay"),
      loadProtectedImage(data.mask_url,"mask")
    ]);
    document.getElementById("vision").textContent=data.vision_answer;
    document.getElementById("result").hidden=false;
    status.textContent="分析完成。";
  }catch(error){status.textContent="错误："+error.message;}
});
</script>
</body></html>"""


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), user: str = Depends(current_user)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件。")

    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with image_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    try:
        result = analyze_image(image_path)
        result["overlay_url"] = (
            f"/api/output/{Path(result['segmentation_overlay']).name}"
        )
        result["mask_url"] = f"/api/output/{Path(result['segmentation_mask']).name}"
        result["record_id"] = save_record(result)
        return result
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/output/{filename}")
def output_file(filename: str, user: str = Depends(current_user)) -> FileResponse:
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在。")
    return FileResponse(path)


@app.get("/api/history")
def history(limit: int = 20, user: str = Depends(current_user)) -> dict:
    return {"items": list_records(limit)}


@app.post("/api/register")
def register(account: Account) -> dict:
    try:
        create_user(account.username, account.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "注册成功，请登录。"}


@app.post("/api/login")
def login(account: Account) -> dict:
    if not authenticate_user(account.username, account.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
    return {"access_token": create_token(account.username), "token_type": "bearer"}
