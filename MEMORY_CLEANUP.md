# SAM3 / SAM3D / Z-Image 空闲显存回收

本仓库接入 2026-09-07 服务器部署中使用的空闲回收逻辑。Qwen 不属于该启动脚本的管理范围。

仓库保留现有源码目录，由启动脚本映射到镜像运行路径；不需要额外维护一份 `service_overrides/`。

## 文件

| 文件 | 用途 |
|---|---|
| `bash_scripts/start_all.sh` | 验证 Python 语法，将托管入口复制到容器；文件有变化时仅重启相应服务，等待 HTTP health 就绪 |
| `source/common/gpu_memory.py` | 共用 CUDA 空闲回收模块 |
| `source/sam3-server/server/inference.py` | SAM3：跟踪请求，返回 CPU 分割结果后进入空闲计时 |
| `source/sam3d-server/server/flask_api.py` | SAM3D：跟踪排队和运行任务，逐对象导出后释放完整推理结果 |
| `source/z-image-server/server/inference.py` | Z-Image：图片编码为 PNG 字节后进入空闲计时 |

应用代码在容器内各自的 `server/` 目录运行；共用模块同样复制到该目录。现有容器普通 `docker restart` 会保留已同步文件。重新创建容器应使用本目录 `bash_scripts/start_all.sh`，以便安装这些覆盖文件；直接使用旧镜像手工启动不会自动包含本次修复。

## 回收策略

请求进入 GPU 工作队列前计数，成功、失败、取消后对应减计数。待处理工作为 0 且连续空闲 **1 秒** 后，后台线程执行垃圾回收、等待本进程 GPU 工作结束、调用 `torch.cuda.empty_cache()`。新请求会取消尚未开始的回收；回收已经开始时，新请求等待回收完成后才进入 GPU 工作。

模型权重保留。回收只归还空闲缓存，不能释放仍被模型或第三方库引用的显存。连续请求之间不会强制清缓存。SAM3D 的原有单 worker、共享 pointmap、坐标转换及纹理烘焙梯度逻辑保持原行为。

SAM3D 每个 mask 的完整推理结果只保留到该对象的 GLB/CPU pose 导出完成，不再积累整批原始 GPU outputs。历史任务及下载资产的保留行为沿用原实现，本次没有添加资产过期删除。

## 验证新请求

由客户端正常调用原有接口，无需添加参数。最后一个任务结束后等待约 1 秒，再查看 GPU 和服务日志：

```bash
nvidia-smi
docker logs --since 10m gensim-sam3-server 2>&1 | grep '\[gpu-memory\]'
docker logs --since 10m gensim-sam3d-server 2>&1 | grep '\[gpu-memory\]'
docker logs --since 10m gensim-z-image-server 2>&1 | grep '\[gpu-memory\]'
```

日志事件 `configured` 表示策略已加载；`idle_reclaim` 包含 allocated/reserved 的前后字节数、released_bytes 和耗时；`reclaim_error` 表示回收异常，需要继续排查。日志不记录图片、提示词、密钥或对象资产。

默认空闲时间由共用模块控制，也支持容器环境变量 `GENSIM_CUDA_IDLE_SECONDS`，必须为有限正数；改变已有容器环境需相应更新部署定义。启动脚本使用默认 1 秒，不会将宿主机的同名环境变量自动传入容器。

## 验证范围

已用不加载模型的测试验证：队列未清空时不回收、取消过期定时器、回收与新请求互斥、异常后恢复、CPU 路径、SAM3D 队列取消/任务失败计数、逐对象原始输出释放和返回字段、下载资产内容，以及部署脚本首次创建、变更重启、重复运行无变更时不重启、语法错误时不触碰容器。

仓库本地回归命令：

```bash
python3 -m unittest discover -s tests -v
```

2026-09-07 服务器上的同源回收实现经客户端请求验证：SAM3 请求结束后服务进程约 3.87 GiB，SAM3D 约 13.33 GiB；SAM3D 最后一次回收释放约 14.07 GiB 缓存。该快照不是显存配额，也不是本地仓库的 GPU 实测值。Z-Image 当时尚无新的推理请求，不能据此声称已验证其实际回收效果。

本地测试不加载模型。仓库版本重新部署后，仍需通过客户端分别验证三个服务的实际推理与回收效果；健康接口就绪不等于完整推理回归。
