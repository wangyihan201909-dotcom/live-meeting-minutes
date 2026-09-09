# 安装与使用

日常使用只有一步：**双击 `启动.bat`**。

第一次要准备两样东西。你已经装了 LM Studio，所以摘要模型这边基本是现成的，真正要装的只有转写服务。

---

## 一、摘要模型：用你现有的 LM Studio

不需要装 llama.cpp——llama.cpp 就在 LM Studio 内部，它自带 OpenAI 兼容服务。

**当前用的是 MiniCPM5-2B Q8_0**（2.68 GB），`config.json` 里 `llm.model_name` 对应的就是 `minicpm5-2b`。两条命令：

```powershell
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" server start --port 1234
```

```powershell
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" load minicpm5-2b -c 8192 --gpu max --parallel 1 -y
```

也可以在 GUI 里做：Developer 标签页 → 加载 MiniCPM5-2B → 打开 Local Server（端口 1234）。

模型 id 不用手抄——程序启动时会自己查 `/v1/models`，配置里的名字对不上就用服务端实际加载的那个。

> 这个兜底有个前提要知道：LM Studio 的 `/v1/models` 列的是**所有已下载**的模型，不只是当前加载的那个。名字对不上时程序取列表第一个，未必就是你加载的。所以配置里的名字还是尽量写对。

### 选型经过：35B → 4B → 2B

一路降下来的，每一步都有实测支撑。这个任务是 schema 约束下的结构化抽取，不吃推理深度，所以小模型的代价比直觉中小得多。

**35B 装不下。** `Qwen3.6-35B-A3B-UD-IQ4_XS` 单独跑没问题，一轮 18.4 秒，但它和语音模型共存不了：

| 配置 | 可用内存 |
|---|---|
| 什么都不加载 | 17.1 GB |
| 35B（parallel 1, ctx 4096, gpu 0.3） | **0.8 GB** |
| 4B + 语音模型同时运行 | **8.7 GB** |

显存只有 8GB，语音模型要占一部分，所以 35B 的 16.5GB 里最多约 5GB 能上显卡，剩下 11.5GB 必须压内存，加上开销共约 16.3GB。内存塞满的后果不是「慢一点」：实测吞吐从 12 tok/s 掉到 **0.53 tok/s**，LM Studio 里发条消息就崩。

**4B 反而更快**（一轮 10.7 秒 vs 18.4 秒），因为它整个在显存里跑，不用每个 token 都从内存搬权重。

**2B 又快了 4 倍。** MiniCPM5-2B 一轮 **2.6 秒**（抽取 1.4s + 合并 0.7s + 概述 0.4s）。摘要模型持续满载时平均每次抽取 1.2 秒，此时两个模型同时在卡上，仍余 4.7 GB 内存 / 2.7 GB 显存。

质量上，抽取和归类没问题——五类要点分类正确，`owner` 能正确填到人名。**但它会漏 `decision`**：「行，那这样定了，我们下周三先灰度百分之十」这种明确拍板，它倾向于揉进相邻的 `action` 条目，而不单独记一条决议。合并去重也偶有瑕疵：同一批要点重复送入时，个别条目会判成 `add` 而非 `merge`，产生重复。

> 以拍板决策为主的会议，建议换回 `qwen3.5-4b`（Q6_K，3.53 GB）——改 `config.json` 里 `llm.model_name` 一行即可。以记待办和风险为主的，2B 够用且快得多。
>
> 量化选 Q8_0 / Q6_K 而不是 Q4：模型本身已经小了，别再叠加激进量化，那是把两种质量损失乘在一起。两三个 GB 对 8GB 显存完全放得下。
>
> `--parallel 1` 也别省。LM Studio 默认开 4 个并行槽位，每个都要独立 KV 缓存，而这条流水线同一时刻只发一个请求。

### 下载模型

从 ModelScope 下，国内快得多（实测 30 MB/s）。**别在 LM Studio 里搜索下载，它走 HuggingFace。**

```powershell
.\.venv\Scripts\modelscope.exe download --model OpenBMB/MiniCPM5-2B-GGUF --include "*Q8_0*" --local_dir "$env:USERPROFILE\.lmstudio\models\OpenBMB\MiniCPM5-2B-GGUF"
```

下到 LM Studio 的模型目录下（`模型目录\<发布方>\<仓库名>\`），它会自动索引，`lms ls` 里能看到就说明认出来了，列出的那个名字就是 `model_name` 该填的值。

> 用 `curl.exe` 直接下也行，但 Windows 自带的 curl 走 Schannel，经代理时常报 `CRYPT_E_NO_REVOCATION_CHECK`（查不到证书吊销列表）。加 `--ssl-no-revoke` 跳过吊销检查，再用 `certutil -hashfile <文件> SHA256` 比对官方哈希把这层保障补回来。加 `-C -` 可以断点续传。
>
> 注意 PowerShell 里必须写 `curl.exe`——裸写 `curl` 是 `Invoke-WebRequest` 的别名，不认 `-L -o` 这些参数。

**换更大的模型前先想清楚 8GB 显存这个硬约束。** 稠密模型每生成一个 token 都要把全部权重过一遍，塞不进显存就得从内存搬，会被带宽焊死。MoE 可以拆开放——注意力、路由器、共享专家留显存，路由专家扔内存交给 CPU，每 token 只激活一小部分；加载时把 **GPU Offload** 拉高，有 *Force MoE expert weights onto CPU* 这类开关就打开。

---

## 二、转写服务：唯一需要新装的

打开 PowerShell：

```powershell
cd "路径\到\live-meeting-minutes"
```

```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1
```

> 报「禁止运行脚本」的话，先跑 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。
> 看到命令行前面出现 `(.venv)` 就对了。

```powershell
pip install -r requirements.txt
```

```powershell
pip install "whisperlivekit[qwen3-streaming]"
```

### 必须把 torch 换成 CUDA 版

**这一步不能省。** 国内 PyPI 镜像默认给的是纯 CPU 轮子，装完 whisperlivekit 启动时会打印 `Accelerator: CPU only`——显卡完全不参与，转写只能靠 CPU 硬扛。

```powershell
pip uninstall -y torch torchaudio
```

```powershell
pip install torch==2.13.0 torchaudio --index-url https://download.pytorch.org/whl/cu130
```

两个坑：

- **必须先 uninstall。** 直接装覆盖不掉——pip 只比对版本号 `2.13.0`，不区分 `+cpu` 和 `+cu130`，会报 "Requirement already satisfied" 然后跳过。
- **必须是 cu130。** 5070 是 Blackwell (sm_120) 需要 cu128 以上，但 cu128 索引最高只到 torch 2.11.0，cu129 只到 2.9.0，只有 cu130 有 2.13.0。清华的 pytorch-wheels 镜像这几个路径都是 404，官方源直连反而通畅。

装完验证：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 语音模型从 ModelScope 下

实测 **hf-mirror.com 从这台机器连不通**，HuggingFace 更不用说，所以不能指望启动时自动下载。手动下到本地：

```powershell
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir .\models\Qwen3-ASR-0.6B
```

约 1.88 GB。`config.json` 里的 `asr.model_path` 已经指向这个目录，启动器会自动加 `--model-path` 参数，之后再也不碰 HuggingFace。

仓库页：[Qwen3-ASR 合集](https://www.modelscope.cn/collections/Qwen/Qwen3-ASR)。

---

## 三、开会

**双击 `启动.bat`**，浏览器会自动打开控制台。

页面上：

1. **「声音来源」下拉框**选一个带 🔊 的

   电脑上有两种声音：你对麦克风说的，和喇叭里放出来的别人说的。带 🔊 的是喇叭（能听到所有人），带 🎤 的是麦克风（只能听到你自己）。**开会要选 🔊。**

   你这台机器上已经验证过：`[10] 扬声器 (Realtek(R) Audio) [Loopback]`，48kHz 立体声，程序会自动转成转写服务要的 16kHz 单声道。默认就会选中它。

   > 想单独确认音频这一环，随时可以跑 `python selftest.py`，它会采 3 秒并报告音量，不依赖任何服务。

2. **确认 LM Studio 的 Local Server 开着**（这个不归启动器管，它只检查）

3. **点「开始」**

   顶上三个灯依次变绿：摘要模型 → 转写服务 → 采集。转写服务首次启动要下模型，可能等几分钟，底部日志区有进度。

4. 左边出现带时间戳的逐句转写，右边逐渐长出纪要

5. **开完点「停止」**，本场会议自动封档

---

## 四、会后：历史与导出

点「开始」就是新建一场会议，「停止」封档。归档在 `meetings/<日期-时间>/`：录音 `audio.wav`、
逐句转写 `transcript.json`、纪要 `minutes.md`。

右上角**「历史」**进归档：

- 搜索框搜的是标题 + **转写原文 + 纪要内容**，记得某人说过某个词就能翻出那场会
- 点开可以**回放录音**，在「纪要 / 转写原文」之间切换
- 点标题可改名，右侧可删除
- 三个导出按钮：`.md`（纪要）、`.txt`（带时间戳的转写）、`.json`（全部）

`live_minutes.md` 仍然实时刷新，适合拿 Markdown 编辑器开着当实时预览；会后要存档就用
`meetings/` 里那份，它带录音和时间戳。

> `meetings/` 装的是真实会议内容，已经在 `.gitignore` 里，不会被提交。

---

## 故障排查

| 现象 | 原因和处理 |
|---|---|
| 双击后闪退 | 没装 Python。装 3.11+，安装时勾选 "Add to PATH" |
| 「连不上 http://127.0.0.1:1234/v1/models」 | LM Studio 没开，或模型没加载，或 Local Server 开关没打开 |
| 「找不到可执行文件 wlk」 | 第二步的 whisperlivekit 没装进 `.venv` |
| 左边一直没文字 | 声音没采到。确认选的是 🔊 那个，确认喇叭真的在响 |
| **转写碎成单字**（「嗯。 那边。 这样。」） | **电平太低**，不是模型问题。确认「自动增益」勾着；用麦克风的话再去 Windows 声音设置把输入音量拉满、开「麦克风加强」。跑 `python selftest.py` 看峰值，说话时到 30% 以上才够 |
| 左边有字右边没纪要 | 看日志。`模型调用失败` 是 LM Studio 那边的问题；`输出不是合法 JSON` 是模型不听话，换个模型 |
| 日志狂刷「摘要落后」 | 模型跟不上说话速度。换更小的模型；或把 `config.json` 里 `summary.max_wait` 提到 60 |
| 历史里没有刚开完的会 | 全程没转写出内容的空会议会被自动丢弃，日志里有「本场没有任何内容，已丢弃」 |
| 录音下下来播不了 | 上次是被强杀的，WAV 头没回填。重新在历史里点开一次会自动修复 |
| 电脑卡死 | 内存不够。模型 + Windows 就吃掉大半个 32GB，开会时关掉 Chrome |

---

## 附：不用 LM Studio 的话

如果你想让启动器自己管摘要模型，把 `config.json` 改成：

```json
"manage": true,
"port": 8080,
"health_path": "/health",
```

然后需要自己装 llama.cpp（[releases](https://github.com/ggml-org/llama.cpp/releases) 里下 Windows 版，**文件名要有 `cu12.8` 或更高**，你的 5070 是 Blackwell 架构，低版本认不出这张卡），并把 `binary` 指向 `llama-server.exe`、`model_path` 指向 gguf 文件。

这条路的好处是页面上的 `n-cpu-moe` 输入框会生效，可以边看日志里的 `eval time` 边调。调法：从 32 每次减 2，减到启动报显存不足退回一档。注意**速度掉落不会报错**，只会悄悄变慢，所以每档都要记数字。

---

## 已知未验证项

**整条链路已在你这台机器上端到端跑通。** 声音从喇叭出来 → loopback 采集 → 重采样 → 中文转写 → 抽取 → 合并 → 纪要，全自动。

- [x] **音频采集** —— 设备 10 loopback，48kHz 2ch → 16kHz 1ch
- [x] **发包节奏** —— 间隔平均 0.500 秒（偏差 ≤1ms），稳态覆盖率 100%
- [x] **CUDA** —— torch 2.13.0+cu130，识别到 RTX 5070 sm_120
- [x] **中文转写** —— 字准确率约 94%，关键词命中 7/8
- [x] **抽取** —— 7.4 秒，要点分类全部正确
- [x] **合并去重** —— 3.2 秒，同一段音频播两遍只产生 5 条而非 10 条
- [x] **纪要输出** —— 五个分组 + 概述齐全
- [x] **启动器与网页控制台** —— 进程托管、状态灯、字段契约

**一轮合计 10.7 秒，45 秒触发窗口余量 34.3 秒。**

中文转写实测对照：

```
原文  ……字段引擎那块，DS 的流程基本跑通了，但是留空字段的逻辑还没处理完……联调
识别  ……四段引擎那块。DS的流程基本跑通了。但是流控字段的逻辑还没处理完……联条
```

错的是「字段→四段」「留空→流控」「联调→联条」。这是 Windows TTS 的机械合成音，真人说话通常更好。

还没验证的：

- [ ] 真实多人会议（目前只测过单人合成语音）
- [ ] 长会议（超过一小时）的稳定性
- [ ] 页面的视觉呈现（验证了数据和 JS 语法，没截图）

### 三个实测踩过的坑

**思考模式必须关掉。** Qwen3.6 默认会先思考几百个 token 才作答，全部进 `reasoning_content`，`content` 是空的。在这台 12 tok/s 的机器上光思考就 48 秒并撞上 token 上限，等于永远拿不到结果。

实测 LM Studio **不认** `/no_think` 提示词，也**不认** `chat_template_kwargs.enable_thinking=false`，只有 `reasoning_effort: "none"` 生效：48 秒失败 → 5.7 秒成功。代码里已固定带上。

**LM Studio 的 response_format 只认 `json_schema` / `text`。** 传 OpenAI 常用的 `json_object` 会直接 HTTP 400。代码改成首选 `json_schema` 并附真实结构约束（约束解码从根上杜绝非法 JSON），遇到 400 按阶梯自动摘特性，换 llama.cpp 或 Ollama 也能跑。

**吞吐约 12 tok/s。** 比预估的 15-25 低，因为 LM Studio 的 `--gpu` 是按层切分，不像 llama.cpp 的 `--n-cpu-moe` 那样只把路由专家挪到 CPU。想再快就走 README 里「不用 LM Studio」那节，用 llama-server 配 `--n-cpu-moe`。当前速度已经够用。

**报错直接发我。**

### 两个实测发现

**WASAPI loopback 在系统静音时完全不产生数据包。** 实测 10 秒里只有前 1.8 秒有数据，覆盖率 16%。程序已经按墙钟节奏补静音块解决——不补的话转写服务的时间轴不会推进，上一句的结尾会一直卡在未确认状态，直到下次有人开口才被冲出来。

**启动到首包约 3.1 秒**，是 WASAPI 打开 loopback 流的一次性延迟，不是卡住了。点「开始」后前几秒没反应是正常的。

> ⚠️ 另外：网上教程常让你给 `wlk` 加 `--qwen3-streaming-tower-checkpoint qfuxa/...` 参数。**千万别加**，那个是纯英文版，加了中文识别率会从 89% 崩到 14%。
