---
title: "AI应用开发框架对比"
date: 2024-09-30T10:25:23+08:00
# weight: 1
# aliases: ["/first"]
tags: ["框架", "AI应用"]
author: "DIZS"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "深入对比当前主流的AI应用开发框架，包括Langchain、LlamaIndex、Haystack和Dify、Flowise、Langflow等，帮助你快速选择合适的工具进行AI应用开发。"
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

随着大模型的爆火，在业务场景中我们听到AI相关的需求越来越多，这时候我们就要面临一个问题，如何选择一个AI框架，让AI需求快速实现进行落地验证？本文将对当前主流的AI应用开发框架进行比较探讨。在深入讨论框架之前，我们需要先理解AI应用如何与现有业务进行整合，如数字员工、AI客服及AI助理等等。由于大模型在训练和部署阶段对硬件资源的要求较高，因此模型与应用通常会实现分离，Model as a Service（MaaS）成为新的AI开发范式。因此，在AI应用开发过程中，也会遵循这一范式，即将AI应用视为一个独立的服务，为下游各种各样的业务提供AI能力。

## AI应用开发框架
简单来说，目前AI应用开发框架的主流发展方向分为重代码开发和低代码开发平台。其中，重代码开发的代表性工具有：

- [Langchain](https://github.com/langchain-ai/langchain): 92.9k star
- [LlamaIndex](https://github.com/run-llama/llama_index): 35.8k star
- [Haystack](https://github.com/deepset-ai/haystack): 16.9k star

其中`Langchain`的社区和功能性最完善，虽然其抽象程度比较高，定制化难度比较大，但是可以作为快速构建AI应用并进行验证的工具。`LlamaIndex` 更加侧重于RAG应用开发，注重检索任务。`Haystack`简单易懂，抽象程度比`Langchain`低，因此定制化难度会低一些，但是目前社区的支持还比较少。

低代码开发平台的代表性工具有：

- [Dify](https://github.com/langgenius/dify): 46.8k star
- [Flowise](https://github.com/FlowiseAI/Flowise): 30.2k star
- [Langflow](https://github.com/langflow-ai/langflow): 30k star

这三个低代码平台都是通过托拉拽的方式进行AI应用开发，同时提供前端页面访问和API访问，因此可以快速与现有的业务场景进行集成。其中`Dify`上手难度最小，但是涉及到定制化或者集成自定义的一些功能时难度最高；

`Flowise`是基于`langchainjs`和`LLamaIndexTS`进行开发，相当于是对`Langchain`和`LlamaIndex`进行封装，并提供一个低代码平台进行`Langchian`或者`LlamaIndex`模块进行组合；

`Langflow`是对`Langchain`进行封装，形成低代码开发平台，其定制化程度非常高，支持在线实时更改节点代码实时生效。

{{< figure src="/images/AI应用开发低代码平台对比.png" width="100%" align="center" title="AI应用开发低代码平台对比" >}}

## 结论
在AI应用开发的初期，建议先选择`Langchain`和`Dify`作为首选工具，以便快速构建AI应用并进行落地验证。当面对某些实际需求时，如果这些框架无法满足，则可以考虑其他框架的实施方案。一方面，当前大模型及AI应用开发尚处于快速迭代阶段，各框架对新功能的支持程度存在差异，因此了解多个框架的特点是非常必要的。另一方面，AI框架和工具的同质化现象较为严重，从一个框架切换到另一个框架相对容易，因此不必过于担忧切换框架的成本问题。
