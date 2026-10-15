# Visual-Relay：双图参考预算实验

本分支在现有 `i2v-A14B` 双流锚点机制上，新增**按视频时间分配的左右参考预算**。它针对 Img1 与 Img2 之间的片段，让左参考权重逐渐下降、右参考权重逐渐上升，同时保留普通视频上下文参与运动生成。

分支：`codex/paired-reference-budget`。基线：[`2affd871`](https://github.com/GordonChen19/Visual-Relay/commit/2affd871f23b01026fb5118bed7b43f91e332ae7)。新功能默认关闭，继续使用现有 Wan2.2-I2V-A14B checkpoint，无需新增训练权重。模型下载、依赖安装和硬件要求见 [Wan2.2 原始说明](Wan2.2/README.md)。

这是推理机制实验：代码约束直接参考通道的份额，**尚不能据此宣称两张图对生成结果的控制能力已经对等**。

## 方法

原有 A 分支覆盖整段视频，B 分支从 Img2 锚点开始覆盖后缀。设 Img2 的临时锚点位于 latent 索引 `a`。对于锚点之前的 Query，本分支将参考分为三组，各自执行一次 attention、分别归一化：

| 组 | Key 来源 | Value 来源 |
| --- | --- | --- |
| 左参考 `L` | A 的首位置 | A 的首位置 |
| 右参考 `R` | A 的临时锚点 `a` | B 的首位置 |
| 视频上下文 `V` | A 的其余普通位置 | A 的其余普通位置 |

视频上下文包含锚点两侧的普通视频位置，排除已经单独读取的两个参考位置，避免直接重复计入。保留现有普通后缀位置的 RoPE 对齐逻辑。**右参考仍是 `K_A(anchor) + V_B(0)`，没有改成 B/B。**

分别计算 `O_L`、`O_R`、`O_V`，再融合：

```text
u = j / a                         # prefix latent 索引 j = 0, ..., a-1
w(u) = u                          # linear
# 或 w(u) = 3*u*u - 2*u*u*u       # smoothstep

O(j) = (1-rho) * O_V(j)
     + rho * ((1-w(u)) * O_L(j) + w(u) * O_R(j))
```

- `rho` 是图片参考的固定总预算，`1-rho` 留给普通视频上下文。
- `w(u)` 是左右参考交接比例，满足 `w(1-u)=1-w(u)`。中点若落在 latent 网格上，左右直接参考系数相同。
- `j` 是**视频 latent 时间位置**，不是去噪采样步数；同一时间位置的所有空间 token 使用相同交接系数。
- 临时锚点不属于 prefix Query，所以实际计算范围没有 `u=1` 的 Query。VAE 时间压缩和临时位置删除也意味着这里不是逐像素、逐输出帧的精确时间约束。

该改动仅替换 prefix attention。临时锚点自身的更新、B 分支计算、主流残差、后缀融合、采样器和解码路径保持现有机制。后缀的 `image_condition_paired_stream_weight` 仍默认 `0.25`，与新增的 `rho` 作用于不同区域。

相比原先一次 prefix attention，新路径执行三次分组 attention，即每次调用增加两次 prefix attention 计算；继续使用现有 attention 内核，不构造完整 attention 分数矩阵。实际显存与耗时需要 GPU 测量。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--image_condition_paired_prefix_reference_budget` | 不设置，关闭新机制 | 有限数值 `[0,1]`，对应 `rho` |
| `--image_condition_paired_prefix_reference_curve` | `linear` | `linear` 或 `smoothstep` |

注意：

- 需同时开启 `--image_condition_paired_anchor_attention`，使用恰好两张图及 `[0,a]` 两个 latent 索引；Img2 必须处于内部，右侧仍要留出片段。
- **省略 budget 参数才是原算法基线。`rho=0` 是仅使用排除两锚点后的普通上下文，并不等于关闭新机制。** `rho=1` 则不直接读取普通上下文组，但残差和其他网络路径仍存在。
- budget 与 `--image_condition_paired_prefix_route_switch_step` 互斥，避免在某个去噪步骤关闭右图通路，破坏显式预算的含义。
- 初次对照建议省略所有 `paired_*fade*` 参数，固定后缀权重，不启用提示词扩展，减少额外变化。
- 本分支没有自适应 `rho`、回退系数 `eta`、去噪进度预算调度或 K/V 来源切换开关。

## 运行示例

安装依赖并准备 checkpoint 后，从仓库根目录进入 `Wan2.2`。将两张输入图放在仓库根目录，替换 checkpoint 路径及 prompt。下面是 Bash 命令；PowerShell 可以将整条 Python 命令写成一行。

```bash
cd Wan2.2
python generate.py \
  --task i2v-A14B \
  --size '832*480' \
  --frame_num 81 \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --image ../Img1.png --image_latent_index 0 \
  --image ../Img2.png --image_latent_index 12 \
  --image_condition_paired_anchor_attention \
  --image_condition_paired_prefix_reference_budget 0.3 \
  --image_condition_paired_prefix_reference_curve linear \
  --base_seed 123 \
  --sample_steps 40 \
  --prompt "A continuous shot with natural motion and a smooth transition." \
  --save_file paired-budget-0.3-linear.mp4
```

`0.3` 仅为演示数值，未经生成实验验证，不代表推荐最优值。制作基线时，保持图片、顺序、索引、prompt、seed 和其他采样设置不变，删除两个 `paired_prefix_reference_*` 参数，并使用不同输出文件名。

## 验证与实验

CPU 单元测试使用小张量和替代 attention 内核检查路由、公式和边界参数；不加载模型权重，不验证 CUDA 内核或完整视频效果。从 `Wan2.2` 目录运行：

```bash
python -X utf8 -m unittest discover -s tests -p 'test_paired_anchor_*.py' -v
```

本分支已完成本地 CPU 验证：**39 项测试通过，无跳过**。其中包含小尺寸随机初始化 DiT 的实际前向计算，验证标准路径和单进程模拟的 sequence-parallel 路径都将参数传到每一层；分布式集合通信使用恒等替代，不等于真实多 GPU / NCCL 验证。

详细测试入口见 [tests/README.md](Wan2.2/tests/README.md)。CPU attention 测试需要 PyTorch；缺少 PyTorch 时跳过，不能将跳过视为已验证。

建议按以下顺序比较，并为多组图片使用多个固定 seed：

| 对照 | 改动 | 检验目标 |
| --- | --- | --- |
| 原算法基线 | 不设置 budget | 原有控制与运动表现 |
| 固定预算 | 固定曲线，改变 `rho` | 参考增强与运动连续性的取舍 |
| 交接曲线 | 固定 `rho`，比较 linear / smoothstep | 视频时间分配的效果 |
| 后续独立实验，未实现 | 比较当前 A/B 与正确对齐 RoPE 的 B/B | 区分预算问题与参考表示问题 |

除了记录参考份额，还应分别扰动 Img1、Img2，检查改动沿视频时间传播的强弱；交换两端并作时间镜像对照时，要处理动作描述、画布及噪声对应关系，相同 seed 本身不保证反转后的噪声匹配。同时检查临近两端的一致性、中间运动、身份保持、重影、静止和突然切换。系数相等是实现性质，不能作为视觉控制相等的实验结论。

## 已知边界

- 两张图的参考特征仍是网络演化后的隐藏表示；组内 attention 总量的预算不等于原始图片的因果贡献或像素贡献。
- Img1 条件输入、主流 Query 和残差、两侧锚点的演化历史以及因果 VAE 仍然不对称。这个改动没有消除全链路的输入角色差异。
- 普通上下文也可能已携带两张图的信息，分组只消除参考位置的直接重复计入。
- Img2 仍对应解码前被删除的临时参考 latent，没有新增严格到达 Img2 的端点约束。
- 增加或均衡直接参考可能改善控制，也可能损害自然运动。当前没有完整模型视频生成结果支持效果提升；需要上述生成实验验证。

实现入口：[attention.py](Wan2.2/wan/modules/attention.py)、[model.py](Wan2.2/wan/modules/model.py)、[image2video.py](Wan2.2/wan/image2video.py)、[generate.py](Wan2.2/generate.py)。
