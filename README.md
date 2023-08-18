# 每个大模型开发者都应该知道的数字

翻译自 https://github.com/ray-project/llm-numbers

谷歌内部流传了一份由传奇工程师 [Jeff Dean](https://en.wikipedia.org/wiki/Jeff_Dean) 整理的文档，名为《[每个工程师都应该知道的数字](http://brenocon.com/dean_perf.html)》。类似的用于粗略计算的数字对大语言模型（LLM）开发人员也非常有用。我们在此分享 Anyscale 使用的这组数字、为什么这些数字很重要、以及如何利用这些数字。

## Github 版本说明

最后更新：2023-05-17

如果你觉得数字准确性有问题，请提交 issue；如果你觉得数字不够全面，请告知我们或者提交 PR。

我们后续将添加关于不同模型每秒词元的统计数据。

## 提示词

### 40-90%[^1]：通过在prompt中添加“请简要回答”可节省的金额

记住大模型服务是按照返回结果中的词元数来收费的，这意味着要求大模型简化回复能为你省很多钱。除了在提示词中添加“简要回答”这样简单的做法，还有很多做法可以尝试，比如：如果你要用 GPT-4 生成 10 个替代方案，或许你只需要 5 个，能省一半钱。


### 1.3:1 -- 每个单词的平均词元数

大模型通过词元（tokens）来处理和生成文本。词元是单词或单词的子部分，例如 “eating” 可能会分为两个词元 “eat” 和 “ing”。一份 750 字的英文文档大约需要 1000 个词元。英语以外的语言中每个单词的词元数取决于该种语言在大模型 embedding 语料库中的重复度。

了解这个数字很重要，因为大部分大模型计费都是以词元数为依据，并且大模型的上下文窗口大小也是根据词元定义的。


## 价格[^2]

虽然价格会变，但是由于目前大模型的运营成本是如此之高，本节的数字至关重要。此处的数据以 OpenAI 为参考，其他提供商（[Anthropic](https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf)、[Cohere](https://cohere.com/pricing)）的价格也大致在这个范围里。


### ~50:1 -- GPT-4 与 GPT-3.5 Turbo的成本比[^3]

这意味着，对于许多应用，使用 GPT-4 来生成高质量的微调数据或自动评估其他模型效果会更好；这些任务可能只需要执行一次，不用频繁地在推理时执行。GPT-3.5-Turbo 比 GPT-4 要便宜大约 50 倍（倍数不是固定的，因为 GPT-4 中提示词、输出的价格不同）。因此你有必要评估一下 GPT-3.5-Turbo 完成任务的程度；例如摘要任务用 GPT-3.5-Turbo 已经绰绰有余。


### 5:1 -- 使用 GPT-3.5-Turbo 与 OpenAI  embedding 生成文本的成本比

该数字表明在向量数据库中查找信息要比使用大模型便宜得多。例如，在神经信息检索系统（neural information retrieval system）中查找“特拉华州的首府是哪里？”的成本比 GPT-3.5-Turbo 便宜约5倍[^4]。。比 GPT-4 更是便宜高达 250 倍！

### 10:1 -- OpenAI embedding 与私有化部署 embedding 服务的成本比 

> 注意：这个数字受负载和 embedding 批量大小的影响，因此将其视为近似值。

在我们的博客中，我们注意到使用 g4dn.4xlarge 实例（按需价格：1.20美元/小时），我们能够使用 Hugging Face 的 SentenceTransformers 每秒输出约 9000 个词元（其质量与 OpenAI embedding 没什么区别）。根据这个速率和节点类型进行一些简单计算，可知私有化部署的 embedding 服务比 OpenAI 便宜 10 倍左右（这里没有考虑网络传输数据的费用）。


### 6:1 -- OpenAI 微调与基础模型 quiry 的成本比

OpenAI 运行微调模型的成本是基础模型的 6 倍。虽然这很昂贵，但这应该是由于基础模型可以同时服务多个用户、而微调模型只能服务单一用户。这也意味着通过提示词工程来改善基础模型多输出结果，要比微调模型性价比高的多。

### 1:1 -- 私有化部署的基础模型与微调模型 quiry 的成本比

如果你私有化部署一个模型，那么为微调模型提供服务的成本与为基础模型提供服务的成本基本上是相同的，因为这些模型具有相同数量的参数。


## 训练和微调


### ~100万美元: 在 1.4 万亿词元上训练 130 亿个参数模型的成本

[LLaMa论文](https://arxiv.org/abs/2302.13971)提到。使用2048个A100 80GB GPU训练了21天才完成LLaMa的训练。我们假设在Red Pajama训练集上训练自己的模型，并估算成本。上述数字是在假设一切顺利（没有崩溃，第一次计算成功等等）的情况下得出。此外，这还需要 2048 个 GPU。大多数公司无法做到这一点（自吹自擂一下：当然，我们在Anyscale可以做到——这是我们的[核心业务](https://www.anyscale.com/blog/training-175b-parameter-language-models-at-1000-gpu-scale-with-alpa-and-ray)！如果您想了解更多信息，请与我们联系）。训练自己的大模型（LLM）是可行的，但成本不低，而且每次运行实际上需要花费数天的时间才能完成。使用预训练模型要便宜得多。


### &lt; 0.001: 微调与从头开始训练的成本比

虽然这个数字有点笼统，但微调的成本几乎可以忽略不计。例如，我们展示了可以花费[大约 7
美元微调 6B
参数模型](https://www.anyscale.com/blog/how-to-fine-tune-and-serve-llms-simply-quickly-and-cost-effectively-using)。即使按 OpenAI 最昂贵的可微调模型 Davinci 的价格计算，每 1000 个标记的成本也是 3 美分。这意味着对莎士比亚的全部作品（约100万字）进行微调，大约仅仅需要40美元[^5]。然而，微调和从头训练存在巨大区别，不能相提并论。


## GPU 显存

如果你正在私有化部署一个模型，了解 GPU 的显存非常重要，因为大模型会极度占用 GPU 的显存。以下统计数据专门针对推理过程。对于训练或微调，需要更多的显存。

### V100: 16GB, A10G: 24GB, A100: 40/80GB: GPU 显存容量

这可能看起来令人费解，但了解不同类型 GPU 的显存非常重要。GPU 的显存大小会限制你的大模型的最大参数量。一般来说， A10G更受欢迎，按照 AWS 按需价格，它的价格为每小时 1.50 至 2 美元，并且具有 24G GPU 显存，而 A100 按 AWS 按需价格，每台价格约为 5 美元。


### 2x 模型参数量: 用于提供大语言模型（LLM）的典型GPU内存要求

举个例子，如果你有一个 70 亿参数的模型，大约需要 14GB 的 GPU 显存空间。这是因为，每个参数通常需要一个16 比特浮点数（或 2 字节）。通常情况下不需要超过 16 比特的精度，而当精度降低到 8 比特时，分辨率开始降低（在某些情况下还是可以接受的）。经过研究，出现了一些可以减少显存需求的方法，例如项目 llama.cpp 用 4 比特量化（8 比特也可行）实现在 6GB 的 GPU 上运行一个 130 亿参数的模型，但这是个例。
~1GB: embedding模型的

### ~1GB: embedding 模型的典型 GPU 显存要求

每当进行句子 embedding（这是一种用于聚类、语义搜索和分类任务的非常典型的操作）时，您需要一个类似[sentence
transformers](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/)的 embedding 模型。OpenAI也提供了自己的商业化的 embedding 模型。

通常不必担心 embedding 在GPU上占用多少显存，它们的需求相当小。甚至可以在同一块GPU上同时使用 embedding 模型和大语言模型。


### >10x: 通过批处理LLM请求的吞吐量提升

通过 GPU 运行大模型 quiry 的延迟非常高，大约在 5 秒钟左右，吞吐量为每秒 0.2 个 quiry 。有趣的是，如果同时运行两个任务，则只需要 5.2  秒。也就是说，如果将 25 个 quiry 并行执行，大约需要 10 秒，吞吐量提高到每秒 2.5 个 quiry 。但是还要考虑下一个数据。


### ~1 MB: 13B 参数模型的 1 个输出词元所需的 GPU 显存

显存大小与生成的最大词元数成正比。例如，如果你想要生成最多 512 个词元（大约 380 个单词），你就需要 512MB 的显存。你可能不以为然。但如果你想要运行更大的批次，显存需求就开始增加；如果要批处理 16 个请求，你需要 8GB 显存。虽然一些在研的技术能解决这个问题，但这目前仍然是一个难题。

# 速查表

![9089216d67b14f4bad7f1af9f6cebc0](https://github.com/NascentCore/llm-numbers-cn/assets/138741722/48f7f00a-4160-4193-a68c-8cb6f65d6a9b)




# 专有名称中英文对照表

|    中文    |   英文    |
| :--------: | :-------: |
|   大模型   |    LLM    |
|   提示词   |  prompt   |
|    词元    |   token   |
|    微调    | fine-tune |

# 下一步

请参阅我们之前关于解决生成式 AI 基础设施以及[将 LangChain 与 Ray 结合使用的](https://www.anyscale.com/blog/llm-open-source-search-engine-langchain-ray)[博客系列](https://www.anyscale.com/blog/ray-common-production-challenges-for-generative-ai-infrastructure)。 

如果您有兴趣了解有关 Ray 的更多信息，请参阅[Ray.io](http://ray.io/)和[Docs.Ray.io](http://docs.ray.io/)。 

要与 Ray 社区建立联系，请加入[Ray Slack](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform)或我们的[讨论论坛](https://discuss.ray.io/)上的 #LLM 。 

如果您对我们用于 ML 训练和服务的 Ray 部署服务感兴趣，请访问[Anyscale.com/Platform](http://www.anyscale.com/platform)并单击“立即尝试”按钮

**Ray Summit 2023**如果你有兴趣了解更多有关 Ray 如何用于构建高性能和可扩展的大模型应用程序以及如何在 Ray 上微调/训练/服务 LLM 的信息，请参加 9 月 18 日至 20 日参加 Ray [Summit](https://raysummit.anyscale.com/)！我们有一群出色的主题演讲者，包括来自 OpenAI 的 John Schulman 和来自 Cohere 的 Aidan Gomez、有关 Ray 的社区和技术演讲以及[针对LLM的实训](https://github.com/ray-project/ray-educational-materials/blob/main/NLP_workloads/Text_generation/LLM_finetuning_and_batch_inference.ipynb)。

<!-- Footnotes themselves at the bottom. -->

## 注:

[^1]: 基于 2023 年 5 月 8 日使用一组提示对 GPT-3.5-Turbo 进行的实验.

[^2]: 2023 年 5 月 8 日检索自http://openai.com/pricing。

[^3]: **GPT-4**: 提示词 6 美分/1000 词元，生成词 12 美分/1000 词元（32,000窗口大小，8,000窗口大小的费用为这个的一半）。**GPT-3.5 Turbo**：0.2美分/1000词元。

[^4]: 这里假设矢量查找是“免费的”。虽然事实并非如此，但它使用
CPU（便宜得多）并且速度相当快。

[^5]: 100 万个单词 / 0.75 个词元/单词 / 1000*0.03 = 40 美元。
