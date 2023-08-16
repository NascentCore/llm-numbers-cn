# 每个 LLM 开发人员都应该知道的数字

谷歌的传奇工程师[Jeff Dean](https://en.wikipedia.org/wiki/Jeff_Dean)整理了一份文件，名为 [Numbers every Engineer should know](http://brenocon.com/dean_perf.html)《每个工程师都应该知道的数字》。对于大语言模型（LLM）开发人员来说，拥有一组类似的数字非常有用，因为这些数字对于粗略计算很有用。在这里，我们分享了 Anyscale 使用的特定数字、为什么该数字很重要以及如何利用它来为您带来优势。

## Github版本注释

最后更新：2023-05-17

如果您认为数字不准确或者不够全面，请提出问题或者上传文档。

我们考虑继续添加关于不同模型每秒词元的统计数据。

## Prompts


### 40-90%[^1]:通过在prompt中添加“回答请简洁”可节省的金额

请务必记住，您是通过词元数来支付费用的。要求大语言模型简洁回复可以节省很多钱。不仅仅只在prompt中添加“简洁”的要求，类似的情况还有更多：如果您想使用 GPT-4 提出 10 个替代方案，也许可以要求它提供 5 个，从而节省一半的钱。


### 1.3:1 -- 每个单词的平均词元数

大语言模型（LLMs）通过词元运作。词元（tokens）是单词或单词的子部分，例如“eating”可能会分为两个词元“eat”和“ing”。一份 750 字的英文文档大约需要 1000 个词元。对于英语以外的语言，每个单词的词元数会根据它们在大语言模型嵌入语料库中的共性而增加。

知道这个比率很重要，因为大多数计费都是以词元为依据，并且 LLM 的上下文窗口大小也是根据词元定义的。


## 价格[^2]

价格固然可能会发生变化，但鉴于大语言模型的运营成本，本节中的数字至关重要。我们使用 OpenAI 来获取此处的数据，其他提供商（[Anthropic](https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf)、[Cohere](https://cohere.com/pricing)）的价格也大致相同。


### ~50:1 -- GPT-4 与 GPT-3.5 Turbo的成本比[^3] 

这意味着对于许多实际应用而言，使用GPT-4来生成高质量的微调数据或自动评估其他模型会更加优越，这些任务可能只需要执行一次，而不是频繁地嵌入在推理周期中。使用GPT-3.5-Turbo相比GPT-4要便宜大约50倍（估计是GPT-4对prompt和生成的输出收费方式不同）。因此您确实需要评估一下您在使用GPT-3.5-Turbo上的需求。对于诸如摘要等任务来说，GPT-3.5-Turbo已经足够了。


### 5:1 -- 使用 GPT-3.5-Turbo 与 OpenAI 嵌入生成文本的成本比

这意味着在向量存储中查找信息要比请求语言模型生成信息便宜得多。例如，在神经信息检索系统中查找“特拉华州的首府是什么？”的成本，比起询问GPT-3.5-Turbo要少约5倍[^4]。与GPT-4相比，成本差距高达250倍！


### 10:1 -- OpenAI 嵌入与自托管嵌入的成本比 

> 注意：这个数字受负载和嵌入批量大小的影响，因此将其视为近似值。

在我们的博客文章中，我们注意到使用g4dn.4xlarge实例（按需价格：1.20美元/小时），我们能够使用Hugging Face的SentenceTransformers每秒嵌入约9000个标记（这几乎与OpenAI的嵌入一样好）。根据这个速率和节点类型进行一些基本数学计算，表明自行托管嵌入要便宜得多（便宜了10倍左右），这还没有考虑传输数据的费用。


### 6:1 -- OpenAI 微调与基础模型查询的成本比

在OpenAI上为微调模型提供服务的成本是为基础模型提供服务成本的6倍。虽然这相当昂贵，但由于基础模型可能具有多租户性质，这可能是有道理的。这也意味着与微调定制模型相比，调整基础模型的提示要更加成本效益。


### 1:1 -- 自托管的基础模型与微调模型查询的成本比率

如果您自行托管一个模型，那么为微调模型提供服务的成本与为基础模型提供服务的成本基本上是相同的：这些模型具有相同数量的参数。


## 训练和微调


### ~100万美元: 在 1.4 万亿词元上训练 130 亿个参数模型的成本

[LLaMa论文](https://arxiv.org/abs/2302.13971)提到他们使用了2048个A100 80GB GPU训练了21天才完成LLaMa的训练。我们曾考虑过在Red Pajama训练集上训练自己的模型，然后我们进行了计算。上述假设一切顺利，没有崩溃，第一次计算成功等等。此外，这还涉及到协调2048个GPU。大多数公司无法做到这一点（自吹自擂一下：当然，我们在Anyscale可以做到——这是我们的[核心业务](https://www.anyscale.com/blog/training-175b-parameter-language-models-at-1000-gpu-scale-with-alpa-and-ray)！如果您想了解更多信息，请与我们联系）。重点是，训练自己的大语言模型（LLM）是可行的，但成本不低，而且每次运行实际上需要花费数天的时间才能完成。使用预训练模型要便宜得多。


### &lt; 0.001: 微调与从头开始训练的成本比率

这有点笼统，但微调的成本几乎可以忽略不计。例如，我们展示了可以花费[大约 7 美元微调 6B 参数模型](https://www.anyscale.com/blog/how-to-fine-tune-and-serve-llms-simply-quickly-and-cost-effectively-using)。即使按OpenAI最昂贵的可微调模型Davinci的价格计算，每1000个标记的成本也是3美分。这意味着对莎士比亚的全部作品（约100万字）进行微调，大约需要40美元[^5]。然而，微调是一回事，从头开始训练是另一回事...


## GPU 显存

如果您正在自托管一个模型，知道GPU内存非常重要，因为大语言模型（LLMs）会极度占用GPU的内存。以下统计数据专门针对推理。对于训练或微调，需要更多的内存。


### V100: 16GB, A10G: 24GB, A100: 40/80GB: GPU 显存容量

这可能看起来很奇怪，但了解不同类型 GPU 拥有的内存量非常重要。这将限制您的大语言模型可以拥有的参数数量。一般来说，我们喜欢使用 A10G，因为按照 AWS 按需价格，它们的价格为每小时 1.50 至 2 美元，并且具有 24G GPU 内存，而 A100 按 AWS 按需价格，每台价格约为 5 美元。


### 2x 模型参数量: 用于提供大语言模型（LLM）的典型GPU内存要求

例如，如果您有一个70亿参数的模型，大约需要14GB的GPU内存空间。这是因为大多数时候，每个参数需要一个16位浮点数（或2字节）。通常情况下，不需要超过16位的精度，而当降低到8位精度时，开始失去分辨率（尽管在某些情况下可能是可以接受的）。当然，有一些办法可以减少这个内存需求，特别是像llama.cpp这样的项目，它通过积极量化降低到4位（甚至8位也没有太大影响）来在6GB的GPU上运行一个130亿参数的模型，但这不是典型的情况。


### ~1GB: 嵌入模型的典型 GPU 显存要求

每当您进行句子嵌入（这是一种用于聚类、语义搜索和分类任务的非常典型的操作）时，您需要一个像[sentence transformers](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/)这样的嵌入模型。OpenAI也提供了自己的商业化嵌入模型。

您通常不必担心嵌入在GPU上占用多少内存，它们相当小。我们甚至在同一块GPU上同时使用了嵌入模型和大语言模型。


### >10x: 通过批处理LLM请求的吞吐量提升

通过GPU运行LLM查询的延迟非常高：可能需要5秒钟，吞吐量为每秒0.2个查询。有趣的是，如果同时运行两个任务，可能只需要5.2秒。这意味着如果您能够将25个查询捆绑在一起，大约需要10秒，吞吐量提高到每秒2.5个查询。但是，请参考下一点。


### ~1 MB: 13B 参数模型的 1 个输出词元所需的 GPU 显存

您所需的内存量与您想要生成的最大词元数成正比。例如，如果您想要生成最多512个词元（大约380个词）的输出，您就需要512MB的内存。您可能会说没什么大不了的——我有24GB的空闲内存，512MB算什么？然而，如果您想要运行更大的批次，内存需求就开始累积。所以，如果您想要批处理16个词元，您就需要8GB的内存空间。虽然一些正在开发的技术可以解决这个问题，但这仍然是一个现实问题。

# 速查表

<img width="1097" alt="Screenshot 2023-05-17 at 1 46 09 PM" src="https://github.com/ray-project/llm-numbers/assets/9677264/5d40c6a3-84d7-436a-8fc4-a8d58008765d">

# 下一步

请参阅我们之前关于解决生成式 AI 基础设施以及[将 LangChain 与 Ray 结合使用的](https://www.anyscale.com/blog/llm-open-source-search-engine-langchain-ray)[博客系列](https://www.anyscale.com/blog/ray-common-production-challenges-for-generative-ai-infrastructure)。 如果您有兴趣了解有关 Ray 的更多信息，请参阅[Ray.io](http://ray.io/)和[Docs.Ray.io](http://docs.ray.io/)。 要与 Ray 社区建立联系，请加入[Ray Slack](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform)或我们的[讨论论坛](https://discuss.ray.io/)上的 #LLM 。 如果您对我们用于 ML 训练和服务的 Ray 托管服务感兴趣，请访问[Anyscale.com/Platform](http://www.anyscale.com/platform)并单击“立即尝试”按钮

**Ray Summit 2023：**如果您有兴趣了解更多有关 Ray 如何用于构建高性能和可扩展的 LLM 应用程序以及如何在 Ray 上微调/训练/服务 LLM 的信息，请参加 9 月 18 日至 20 日参加 Ray [Summit](https://raysummit.anyscale.com/)！我们有一组出色的主题演讲者，包括来自 OpenAI 的 John Schulman 和来自 Cohere 的 Aidan Gomez、有关 Ray 的社区和技术演讲以及[针对法学硕士的实践培训](https://github.com/ray-project/ray-educational-materials/blob/main/NLP_workloads/Text_generation/LLM_finetuning_and_batch_inference.ipynb)。

<!-- Footnotes themselves at the bottom. -->

## 注:

[^1]:
     基于 2023 年 5 月 8 日使用一组提示对 GPT-3.5-Turbo 进行的实验.

[^2]:
     2023 年 5 月 8 日检索自http://openai.com/pricing。

[^3]:
    **GPT-4**: 提示部分6美分/每1000个标记，生成部分12美分/每1000个标记（32,000窗口版本，8,000窗口版本为这个的一半）。**GPT-3.5 Turbo**: 提供0.2美分/每1000个标记。

[^4]:
     这假设矢量查找是“免费的”。事实并非如此，但它使用 CPU（便宜得多）并且速度相当快。

[^5]:
    100 万个单词 / 0.75 个词元/单词 / 1000*0.03 = 40 美元。
