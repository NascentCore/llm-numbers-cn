# 每个大模型开发者都应该知道的数字

翻译自 https://github.com/ray-project/llm-numbers

谷歌内部流传了一份由传奇工程师 [Jeff Dean](https://en.wikipedia.org/wiki/Jeff_Dean) 整理的文档，名为《[每个工程师都应该知道的数字](http://brenocon.com/dean_perf.html)》。大语言模型（LLM）开发人员们同样需要一组类似的数字为粗略计算做参考。我们在此分享 Anyscale 使用的一组数字、并介绍为什么这些数字很重要、以及如何利用这些数字。

## Github 版本说明

最后更新：2023-05-17

若您发现文中的数字存在不准确之处，或希望补充更多数据，请提交相应的 issue 或 PR 以通知我们。

我们将在后续补充不同模型推理速度（token/s）的相关统计数据。

## 提示词

### 40-90%[^1]：通过在prompt中添加“请简要回答”可节省的金额

大模型服务通常按照返回结果中的词元数来收费，因此要求大模型精简回复能为你省下很多钱。除了在提示词中添加“简要回答”这样简单的做法，还有很多做法可以尝试，比如：GPT-4为你的提问自动生成了 10 条答案，如果明确要求它只生成 5 条，就可以节省一半的费用。


### 1.3:1 -- 每个单词的平均词元数

大模型通过词元（tokens）来处理和生成文本。词元是单词或单词的子部分，例如 “eating” 可能会分为两个词元 “eat” 和 “ing”。一份 750 字的英文文档大约需要 1000 个词元。对于非英语的语言，单词被切分成的词元数量可能会增加，这具体取决于该语言在大型模型 embedding 语料库中出现的频率。

了解这个数字很重要，因为大部分大模型计费都是以词元数为依据，并且大模型的上下文窗口大小也是根据词元数定义的。


## 价格[^2]

尽管大模型服务的价格有所波动，但考虑到大模型的运行成本普遍较高，本节所提供的数据显得尤为关键。此处的数据以 OpenAI 为参考，其他提供商（[Anthropic](https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf)、[Cohere](https://cohere.com/pricing)）的价格也大致在这个范围里。


### ~50:1 -- GPT-4 与 GPT-3.5 Turbo的成本比[^3]

对于很多应用场景，这个数字意味着使用GPT-4来执行那些只需执行一次、不需要在推理时频繁执行的任务更为合适，例如生成高质量的微调数据或自动评估其他模型的效果；GPT-3.5-Turbo 比 GPT-4 要便宜大约 50 倍（倍数不是固定的，因为 GPT-4 中提示词、输出词的价格不同）。因此你有必要评估一下 GPT-3.5-Turbo 完成任务的效果，例如摘要任务使用 GPT-3.5-Turbo 已经绰绰有余。


### 5:1 -- 使用 GPT-3.5-Turbo 与 OpenAI  embedding 生成文本的成本比

该数字表明在向量数据库中查找信息要比使用大模型便宜得多。例如，在神经信息检索系统（neural information retrieval system）中查找“特拉华州的首府是哪里？”的成本比 GPT-3.5-Turbo 便宜约5倍[^4]，比 GPT-4 更是便宜高达 250 倍！

### 10:1 -- OpenAI embedding 与私有化部署 embedding 服务的成本比 

> 注意：这个数值会受到负载和嵌入批量大小的影响，所以请将其看作是一个近似估计。

在我们的博客中，我们在 g4dn.4xlarge 实例（按需价格：1.20美元/小时）上使用 Hugging Face 的 SentenceTransformers 进行推理，每秒可以输出约 9000 个词元（其质量与 OpenAI embedding 没什么区别）。根据这个速率和实例类型进行一些简单计算，可知私有化部署的 embedding 服务比 OpenAI 便宜 10 倍左右（此处未考虑网络数据传输的成本）。


### 6:1 -- 调用 OpenAI 微调模型与基础模型的成本比

OpenAI 运行微调模型的成本是基础模型的 6 倍。尽管这听起来价格不菲，但实际上是合理的，因为基础模型可以同时服务多个用户、而微调模型只能服务单一用户。这也意味着通过提示词工程来改善基础模型的输出结果，要比微调模型性价比高的多。

### 1:1 -- 调用私有化部署的基础模型与微调模型的成本比

如果你私有化部署一个模型，那么为微调模型提供服务的成本与为基础模型提供服务的成本基本上是相同的，因为这些模型具有相同数量的参数量。


## 训练和微调


### ~100万美元: 在 1.4 万亿词元上训练 130 亿个参数模型的成本

[LLaMa论文](https://arxiv.org/abs/2302.13971)提到，其使用 2048 个 A100 80GB GPU 训练了 21 天才完成LLaMa的训练。我们假设在Red Pajama训练集上训练自己的模型，并估算成本。上述数字是在假设一切顺利（没有崩溃，训练一次性成功等等）的情况下得出。此外，这还需要 2048 个 GPU。大多数公司无法做到这一点（自吹自擂一下：这在Anyscale可以轻松做到——这是我们的[核心业务](https://www.anyscale.com/blog/training-175b-parameter-language-models-at-1000-gpu-scale-with-alpa-and-ray)！如果您想了解更多信息，请与我们联系）。训练自己的大模型（LLM）是可行的，但成本不低，而且每次运行实际上需要花费数天的时间才能完成。直接使用预训练模型要便宜得多。


### &lt; 0.001: 微调与从头开始训练的成本比

虽然这个数字有点笼统，但微调的成本几乎可以忽略不计。例如，我们展示了可以花费[大约 7
美元微调 6B
参数模型](https://www.anyscale.com/blog/how-to-fine-tune-and-serve-llms-simply-quickly-and-cost-effectively-using)。即使按 OpenAI 最昂贵的可微调模型 Davinci 的价格计算，每 1000 个词元的成本也是 3 美分。这意味着对莎士比亚的全部作品（约100万字）进行微调，大约仅仅需要40美元[^5]。


## GPU 显存

如果您正在进行模型的私有部署，了解GPU的显存尤为关键，因为大型模型会大量消耗GPU显存。以下的数据主要针对推理阶段；而对于训练或微调，显存需求将会更高。

### V100: 16GB, A10G: 24GB, A100: 40/80GB: GPU 显存容量

这可能看起来令人费解，但了解不同类型 GPU 的显存非常重要。GPU 的显存大小会限制你的大模型的最大参数量。一般来说，A10G更受欢迎：按照 AWS 按需价格，它的价格为每小时 1.50 至 2 美元，并且具有 24G GPU 显存；而 A100 按 AWS 按需价格，每台价格约为 5 美元。


### 2x 模型参数量: 大语言模型（LLM）推理的典型GPU内存要求

举个例子，一个 70 亿参数的模型大约需要 14GB 的 GPU 显存空间。这是因为，每个参数通常需要一个 16 比特浮点数（2 字节）来保存。模型通常情况下不需要超过 16 比特的精度，然而当精度降低到 8 比特时，模型的生成质量开始下降（在某些情况下还是可以接受的）。当然，有很多可以减少显存需求的方法，例如项目 llama.cpp 大胆地使用 4 比特量化实现在 6GB 的 GPU 上运行一个 130 亿参数的模型（或 8 比特以保证生成质量），但这仅仅是个例。

### ~1GB: embedding 模型的典型 GPU 显存要求

每当进行句子 embedding（这是一种用于聚类、语义搜索和分类任务的非常典型的操作）时，您需要一个类似[sentence
transformers](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/)的 embedding 模型。OpenAI也提供了自己的商业化的 embedding 模型。

通常，我们无需过于担忧 embedding 在GPU上的显存占用，其需求非常有限。我们甚至会将 embedding 模型和大型语言模型放在同一块GPU上运行。


### >10x: 通过批处理LLM请求的吞吐量提升

通过 GPU 进行大模型推理的延迟非常高，例如，处理一条查询可能需要5秒钟，其吞吐量是每秒0.2个查询。但有趣的现象是，如果你同时处理两个查询，可能只需5.2秒。这意味着，如果你能够将25个查询集中并行处理，它们总共可能只需要10秒，此时的吞吐量就提高到了每秒2.5个查询。然而，你还需要考虑下面这个数据：


### ~1 MB: 13B 参数模型的 1 个输出词元所需的 GPU 显存

显存需求与生成的最大词元数成正比。例如，你至少需要 512MB 的显存来生成 512 个词元（大约 380 个单词）。相对于GPU的显存，512MB可能并不显得很大，但当你希望处理更大的批次时，显存的需求会逐渐增加。例如，若要并行处理16个请求，你就需要8GB（16 * 512MB）的显存。尽管有些正在研究中的技术可以应对此问题，但目前这依然是个挑战。

# 速查表

![4dbe44b9656b5fc7299d37dde478973](https://github.com/NascentCore/llm-numbers-cn/blob/main/images/Cheatsheet.png)



# 专有名称中英文对照表

|    中文    |   英文    |
| :--------: | :-------: |
|   大模型   |    LLM    |
|   提示词   |  prompt   |
|    词元    |   token   |
|    微调    | fine-tune |

# 下一步

请参阅我们之前关于构建生成式 AI 基础设施以及[将 LangChain 与 Ray 结合使用的](https://www.anyscale.com/blog/llm-open-source-search-engine-langchain-ray)[系列博客](https://www.anyscale.com/blog/ray-common-production-challenges-for-generative-ai-infrastructure)。 

如果您有兴趣了解有关 Ray 的更多信息，请参阅[Ray 项目主页](http://ray.io/)以及[Ray 官方文档](http://docs.ray.io/)。 

要与 Ray 社区建立联系，请加入[Ray Slack](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform)或我们的[社区论坛](https://discuss.ray.io/)上的 #LLM 专区。 

如果您对我们用于 ML 训练和服务的 Ray 部署服务感兴趣，请访问[Anyscale.com/Platform](http://www.anyscale.com/platform)并单击“立即尝试”按钮

**Ray Summit 2023**：如果你有兴趣了解更多有关 Ray 如何用于构建高性能和可扩展的大模型应用程序，以及如何在 Ray 上微调/训练/服务 LLM 的信息，请参加9月18至20日的 Ray [Summit](https://raysummit.anyscale.com/)！我们邀请了一群出色的演讲嘉宾，包括来自 OpenAI 的 John Schulman 和来自 Cohere 的 Aidan Gomez，带来有关 Ray 的社区发展和技术演讲。届时我们还会开展[LLM 微调与推理实战训练](https://github.com/ray-project/ray-educational-materials/blob/main/NLP_workloads/Text_generation/LLM_finetuning_and_batch_inference.ipynb)。

<!-- Footnotes themselves at the bottom. -->

## 注:

[^1]: 基于 2023 年 5 月 8 日使用一组提示对 GPT-3.5-Turbo 进行的实验.

[^2]: 2023 年 5 月 8 日检索自http://openai.com/pricing。

[^3]: **GPT-4**: 提示词 $0.06/1K tokens，生成词 $0.12/1K tokens（此为 32,000 窗口大小模型的定价，8,000窗口大小的费用为这个的一半）。**GPT-3.5 Turbo**：$0.002/1K tokens。

[^4]: 这里假设矢量查找是“免费的”。虽然事实并非如此，但它使用 CPU（便宜得多）并且速度相当快。

[^5]: 100 万个单词 / 0.75 个词元/单词 / 1000*0.03 = 40 美元。
