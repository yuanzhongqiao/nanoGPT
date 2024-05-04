<div class="Box-sc-g0xbh4-0 bJMeLZ js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">纳米GPT</font></font></h1><a id="user-content-nanogpt" class="anchor" aria-label="永久链接：nanoGPT" href="#nanogpt"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/karpathy/nanoGPT/blob/master/assets/nanogpt.jpg"><img src="https://github.com/karpathy/nanoGPT/raw/master/assets/nanogpt.jpg" alt="纳米GPT" style="max-width: 100%;"></a></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于训练/微调中型 GPT 的最简单、最快的存储库。它是</font></font><a href="https://github.com/karpathy/minGPT"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">minGPT</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">的重写</font><font style="vertical-align: inherit;">，优先考虑牙齿而不是教育。仍在积极开发中，但目前该文件</font></font><code>train.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在 OpenWebText 上重现了 GPT-2 (124M)，在大约 4 天的训练中在单个 8XA100 40GB 节点上运行。代码本身简单易读：</font></font><code>train.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">是一个约 300 行样板训练循环和约</font></font><code>model.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">300 行 GPT 模型定义，可以选择从 OpenAI 加载 GPT-2 权重。就是这样。</font></font></p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="https://github.com/karpathy/nanoGPT/blob/master/assets/gpt2_124M_loss.png"><img src="https://github.com/karpathy/nanoGPT/raw/master/assets/gpt2_124M_loss.png" alt="复制124米" style="max-width: 100%;"></a></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">因为代码非常简单，所以很容易满足您的需求，从头开始训练新模型，或微调预训练检查点（例如，当前可用的最大起点是 OpenAI 的 GPT-2 1.3B 模型）。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">安装</font></font></h2><a id="user-content-install" class="anchor" aria-label="永久链接：安装" href="#install"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>pip install torch numpy transformers datasets tiktoken wandb tqdm
</code></pre><div class="zeroclipboard-container">
  
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">依赖项：</font></font></p>
<ul dir="auto">
<li><a href="https://pytorch.org" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">火炬</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">&lt;3</font></font></li>
<li><a href="https://numpy.org/install/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">numpy</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;"> &lt;3</font></font></li>
<li><code>transformers</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对于拥抱面变压器&lt;3（加载GPT-2检查点）</font></font></li>
<li><code>datasets</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对于huggingface数据集&lt;3（如果你想下载+预处理OpenWebText）</font></font></li>
<li><code>tiktoken</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对于 OpenAI 的快速 BPE 代码 &lt;3</font></font></li>
<li><code>wandb</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于可选日志记录 &lt;3</font></font></li>
<li><code>tqdm</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对于进度条&lt;3</font></font></li>
</ul>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">快速开始</font></font></h2><a id="user-content-quick-start" class="anchor" aria-label="永久链接：快速启动" href="#quick-start"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果你不是深度学习专业人士，而只是想感受其中的魔力并上手，那么最快的入门方法就是在莎士比亚的作品上训练角色级的 GPT。首先，我们将其作为单个 (1MB) 文件下载，并将其从原始文本转换为一个大的整数流：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python data/shakespeare_char/prepare.py
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这会</font><font style="vertical-align: inherit;">在该数据目录中创建一个</font></font><code>train.bin</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">and 。</font></font><code>val.bin</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">现在是时候训练你的 GPT 了。它的大小很大程度上取决于系统的计算资源：</font></font></p>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我有一个 GPU</font></font></strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。太棒了，我们可以使用</font></font><a href="/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">config/train_shakespeare_char.py</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">配置文件中提供的设置快速训练一个婴儿 GPT </font><font style="vertical-align: inherit;">：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python train.py config/train_shakespeare_char.py
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果你仔细看一下它的内部，你会发现我们正在训练一个上下文大小最多为 256 个字符、384 个特征通道的 GPT，它是一个 6 层 Transformer，每层有 6 个头。在一个 A100 GPU 上，此训练运行大约需要 3 分钟，最佳验证损失为 1.4697。根据配置，模型检查点将被写入</font></font><code>--out_dir</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录中</font></font><code>out-shakespeare-char</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。因此，一旦训练完成，我们就可以通过将采样脚本指向此目录来从最佳模型中进行采样：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python sample.py --out_dir=out-shakespeare-char
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这会生成一些样本，例如：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">哈哈  </font></font><code>¯\_(ツ)_/¯</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。在 GPU 上训练 3 分钟后，对于字符级模型来说已经不错了。通过在此数据集上微调预训练的 GPT-2 模型，很可能可以获得更好的结果（请参阅后面的微调部分）。</font></font></p>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我只有一台 MacBook</font></font></strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（或其他便宜的电脑）。不用担心，我们仍然可以训练 GPT，但我们想把事情降低一个档次。我建议每晚获取最先进的 PyTorch（安装时</font></font><a href="https://pytorch.org/get-started/locally/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在此处选择它</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">），因为它目前很可能使您的代码更加高效。但即使没有它，简单的火车运行也可能如下所示：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在这里，由于我们在 CPU 而不是 GPU 上运行，因此我们必须设置两者</font></font><code>--device=cpu</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，并关闭 PyTorch 2.0 使用</font></font><code>--compile=False</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">.然后，当我们评估时，我们得到的噪声更大，但估计速度更快（</font></font><code>--eval_iters=20</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，从 200 下降），我们的上下文大小只有 64 个字符，而不是 256 个，并且每次迭代的批量大小只有 12 个示例，而不是 64 个。我们还将使用更小的 Transformer（4 层、4 个头、128 个嵌入大小），并将迭代次数减少到 2000（相应地，通常将学习率衰减到 max_iters 左右</font></font><code>--lr_decay_iters</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">）。因为我们的网络很小，所以我们也放松了正则化（</font></font><code>--dropout=0.0</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">）。这仍然需要约 3 分钟的时间，但我们只损失了 1.88，因此样本也更差，但它仍然很有趣：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python sample.py --out_dir=out-shakespeare-char --device=cpu
</code></pre><div class="zeroclipboard-container">
  
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">生成这样的样本：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在 CPU 上运行大约 3 分钟还不错，可以提示正确的角色格式塔。如果您愿意等待更长时间，请随意调整超参数、增加网络大小、上下文长度 ( </font></font><code>--block_size</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">)、训练长度等。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">最后，在 Apple Silicon Macbook 和最新的 PyTorch 版本上，请确保添加</font></font><code>--device=mps</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（“Metal Performance Shaders”的缩写）；然后，PyTorch 使用片上 GPU，可以</font></font><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">显着</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">加速训练（2-3 倍）并允许您使用更大的网络。</font><font style="vertical-align: inherit;">更多信息</font><font style="vertical-align: inherit;">请参见</font></font><a href="https://github.com/karpathy/nanoGPT/issues/28" data-hovercard-type="issue" data-hovercard-url="/karpathy/nanoGPT/issues/28/hovercard"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">第 28 期。</font></font></a><font style="vertical-align: inherit;"></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">复制GPT-2</font></font></h2><a id="user-content-reproducing-gpt-2" class="anchor" aria-label="永久链接：复制 GPT-2" href="#reproducing-gpt-2"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">更认真的深度学习专业人士可能对重现 GPT-2 结果更感兴趣。所以我们开始 - 我们首先对数据集进行标记，在本例中是</font></font><a href="https://openwebtext2.readthedocs.io/en/latest/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">OpenWebText</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，是 OpenAI 的（私有）WebText 的开放复制品：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python data/openwebtext/prepare.py
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这将下载并标记</font></font><a href="https://huggingface.co/datasets/openwebtext" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">OpenWebText</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">数据集。它将创建一个</font></font><code>train.bin</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">and ，</font></font><code>val.bin</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">其中以一个序列保存 GPT2 BPE 令牌 ID，并存储为原始 uint16 字节。然后我们准备开始训练。要重现 GPT-2 (124M)，您至少需要一个 8X A100 40GB 节点并运行：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这将使用 PyTorch 分布式数据并行 (DDP) 运行大约 4 天，损失降至约 2.85。现在，刚刚在 OWT 上评估的 GPT-2 模型的 val 损失约为 3.11，但如果对其进行微调，它将降至约 2.85 区域（由于明显的域差距），从而使两个模型〜匹配。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果您处于集群环境中并且拥有多个 GPU 节点，您可以使 GPU 在 2 个节点上运行，例如：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对互连进行基准测试（例如 iperf3）是个好主意。特别是，如果您没有 Infiniband，那么还要预先考虑</font></font><code>NCCL_IB_DISABLE=1</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">上述启动。您的多节点训练将会起作用，但很可能是</font></font><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">爬行</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。默认情况下，检查点会定期写入</font></font><code>--out_dir</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">.我们可以简单地从模型中采样</font></font><code>$ python sample.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">最后，要在单个 GPU 上进行训练，只需运行</font></font><code>$ python train.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">脚本即可。看看它的所有参数，该脚本试图变得非常可读、可破解且透明。您很可能希望根据您的需要调整其中一些变量。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">基线</font></font></h2><a id="user-content-baselines" class="anchor" aria-label="永久链接：基线" href="#baselines"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">OpenAI GPT-2 检查点使我们能够为 openwebtext 制定一些基线。我们可以得到如下数字：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">并观察 train 和 val 上的以下损失：</font></font></p>
<table>
<thead>
<tr>
<th><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">模型</font></font></th>
<th><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">参数</font></font></th>
<th><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">列车损失</font></font></th>
<th><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">价值损失</font></font></th>
</tr>
</thead>
<tbody>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">总蛋白2</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">124M</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">3.11</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">3.12</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">gpt2-中</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">350M</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2.85</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2.84</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">gpt2-大</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">774M</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2.66</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2.67</font></font></td>
</tr>
<tr>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">gpt2-xl</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">1558M</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2.56</font></font></td>
<td><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2.54</font></font></td>
</tr>
</tbody>
</table>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">然而，我们必须注意，GPT-2 是在（封闭的，从未发布的）WebText 上进行训练的，而 OpenWebText 只是该数据集的尽力开放复制。这意味着存在数据集域差距。事实上，直接在 OWT 上采用 GPT-2 (124M) 检查点并进行微调一段时间，可以将损失降至约 2.85。这将成为更合适的再现基线。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">微调</font></font></h2><a id="user-content-finetuning" class="anchor" aria-label="永久链接：微调" href="#finetuning"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">微调与训练没有什么不同，我们只是确保从预训练模型进行初始化并以较小的学习率进行训练。有关如何在新文本上微调 GPT 的示例，请转至</font></font><code>data/shakespeare</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">并运行</font></font><code>prepare.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">以下载小型莎士比亚数据集，并</font><font style="vertical-align: inherit;">使用 GPT-2 中的 OpenAI BPE 分词器将其渲染为 a</font></font><code>train.bin</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">和。</font></font><code>val.bin</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">与 OpenWebText 不同，这将在几秒钟内运行。微调可能需要很少的时间，例如在单个 GPU 上只需几分钟。运行一个微调示例，例如：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python train.py config/finetune_shakespeare.py
</code></pre><div class="zeroclipboard-container">
 
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这将加载配置参数覆盖</font></font><code>config/finetune_shakespeare.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（尽管我没有对它们进行太多调整）。基本上，我们从 GPT2 检查点进行初始化</font></font><code>init_from</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">并正常训练，只是时间较短且学习率较小。如果内存不足，请尝试减小模型大小（它们是</font></font><code>{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">）或可能减小</font></font><code>block_size</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（上下文长度）。最佳检查点（最低验证损失）将位于</font></font><code>out_dir</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录中，例如</font></font><code>out-shakespeare</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">默认情况下，根据配置文件。然后您可以在以下位置运行代码</font></font><code>sample.py --out_dir=out-shakespeare</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
</code></pre><div class="zeroclipboard-container">
     
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">哇哦，GPT，进入那边某个黑暗的地方。我并没有对配置中的超参数进行太多调整，请随意尝试！</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">采样/推理</font></font></h2><a id="user-content-sampling--inference" class="anchor" aria-label="永久链接：采样/推理" href="#sampling--inference"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">使用该脚本</font></font><code>sample.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">从 OpenAI 发布的预训练 GPT-2 模型或您自己训练的模型中进行采样。例如，以下是从最大可用</font></font><code>gpt2-xl</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">模型中采样的方法：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
</code></pre><div class="zeroclipboard-container">
   
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果您想从您训练的模型中进行采样，请使用</font></font><code>--out_dir</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">适当地指向代码。您还可以使用文件中的一些文本提示模型，例如</font></font><code>$ python sample.py --start=FILE:prompt.txt</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">.</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">效率笔记</font></font></h2><a id="user-content-efficiency-notes" class="anchor" aria-label="永久链接：效率笔记" href="#efficiency-notes"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">对于简单的模型基准测试和分析，</font></font><code>bench.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">可能会有用。它与 的训练循环的核心内容相同</font></font><code>train.py</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">，但省略了许多其他复杂性。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">请注意，代码默认使用</font></font><a href="https://pytorch.org/get-started/pytorch-2.0/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">PyTorch 2.0</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。截至撰写本文时（2022 年 12 月 29 日），该功能已</font></font><code>torch.compile()</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在夜间版本中提供。一行代码的改进是显而易见的，例如，将迭代时间从 ~250ms/iter 减少到 135ms/iter。 PyTorch 团队干得好！</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">待办事项</font></font></h2><a id="user-content-todos" class="anchor" aria-label="永久链接： 待办事项" href="#todos"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">研究并添加 FSDP 而不是 DDP</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在标准评估上评估零样本困惑（例如 LAMBADA？HELM？等）</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Finetune微调脚本，我认为hyperparams不是很好</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">训练期间线性批量大小增加的时间表</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">合并其他嵌入（旋转、不在场证明）</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我认为将检查点中的优化缓冲区与模型参数分开</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">关于网络健康状况的附加日志记录（例如梯度剪辑事件、幅度）</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">关于更好的初始化等的更多调查。</font></font></li>
</ul>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">故障排除</font></font></h2><a id="user-content-troubleshooting" class="anchor" aria-label="永久链接：故障排除" href="#troubleshooting"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">请注意，默认情况下此存储库使用 PyTorch 2.0（即</font></font><code>torch.compile</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">）。这是相当新的和实验性的，并且尚未在所有平台（例如Windows）上可用。如果您遇到相关错误消息，请尝试通过添加</font></font><code>--compile=False</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">标志来禁用此功能。这会减慢代码速度，但至少它会运行。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">有关此存储库、GPT 和语言建模的一些背景信息，观看我的</font></font><a href="https://karpathy.ai/zero-to-hero.html" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">从零到英雄系列</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">可能会有所帮助。具体来说，</font><font style="vertical-align: inherit;">如果您有一些先前的语言建模背景，</font></font><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">GPT 视频会很受欢迎。</font></font></a><font style="vertical-align: inherit;"></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如需更多问题/讨论，请随时访问</font><font style="vertical-align: inherit;">Discord 上的</font></font><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">#nanoGPT ：</font></font></strong><font style="vertical-align: inherit;"></font></p>
<p dir="auto"><a href="https://discord.gg/3zy8kqD9Cp" rel="nofollow"><img src="https://camo.githubusercontent.com/f3b057ade47ec925300af6567ec645a7a1178b1a823f6285daf317cd42cb1fb9/68747470733a2f2f646362616467652e76657263656c2e6170702f6170692f7365727665722f337a79386b71443943703f636f6d706163743d74727565267374796c653d666c6174" alt="" data-canonical-src="https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&amp;style=flat" style="max-width: 100%;"></a></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">致谢</font></font></h2><a id="user-content-acknowledgements" class="anchor" aria-label="永久链接：致谢" href="#acknowledgements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">所有 nanoGPT 实验均由我最喜欢的云 GPU 提供商</font></font><a href="https://lambdalabs.com" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Lambda 实验室</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">的 GPU 提供支持。</font><font style="vertical-align: inherit;">感谢 Lambda 实验室赞助 nanoGPT！</font></font></p>
</article></div>
