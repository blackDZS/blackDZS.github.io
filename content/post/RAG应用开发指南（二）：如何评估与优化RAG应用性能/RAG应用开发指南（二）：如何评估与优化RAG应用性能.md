---
title: "RAG应用开发指南（二）：如何评估与优化RAG应用性能"
description: 
date: 2024-09-29T17:17:35+08:00
image: 
math: true
license: 
hidden: false
comments: true
draft: false
---

## 1. 介绍
在[深入解析RAG原理与实现](/posts/rag应用开发指南一深入解析rag原理与实现/ "深入解析RAG原理与实现")一文中，我们探讨了如何构建RAG应用。然而，在成功构建RAG应用后，新的挑战随之而来：如何评估其性能？我们需要什么样的量化指标来进行有效的评估？

## 2. RAG评估

### 2.1 评估指标
在RAG流程中，主要包括三个核心部分：问题（Query）、检索到的文档（Context）以及模型生成的答案（Answer）。在评估过程中，我们还需要真实答案（Ground Truth）作为基准。在RAG应用中，我们关注两个关键点：其一是检索到的文档（Context），其二是基于检索到的文档所生成的答案（Answer）。下图1展示了这两个部分设置的评估指标，其中左侧列出了与`Answer`相关的指标，右侧则呈现了与`Context`相关的指标。指标计算方法可以参考[RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html)

- **Context**
    - Context Relevance: 评估Context与Query的相关性
    - Context Utilization: 根据Context和Answer计算Context利用率
    - Context Precision: 根据Context和Ground Truth计算检索到的Context准确率
    - Context Recall: 根据Context和Ground Truth计算检索到的Context召回率
    - Context Entities Recall: 根据Context和Ground Truth计算检索到的Context 中Entities的召回率

- **Answer**
    - Answer Relevance: 根据Query和Answer计算Answer与Query的相关性(使用Embedding向量计算)
    - Answer Faithfulness: 根据Context和Answer计算Answer是否来源于Context
    - Answer Semantic Similarity: 根据Ground Truth和Answer计算Answer与Ground Truth的语义相似性(使用Embedding向量计算)
    - Answer Correctness: 根据Ground Truth和Answer计算Answer准确率(使用LLM判断)

通过以上评估指标，我们能够更全面地评价RAG系统的性能。

{{< figure src="/images/RAG Metrics.png" width="100%" align="center" title="图 1. RAG 评估指标" >}}

### 2.2 评估过程
在评估过程中，我们需要对一些标准化的指标进行量化。例如，在计算“上下文精确度”（Context Precision）时，需要评估检索到的上下文对生成答案的有效性。传统的自然语言分析方法在这一指标的评估上存在一定的局限性，但我们可以利用大型语言模型来辅助判断。下图2展示了在RAGAS中关于“上下文精确度”计算过程的Prompt设计。
{{< figure src="/images/Context Precision.png" width="100%" align="center" title="图 2. RAGAS Context Precision Prompt" >}}


## 引用
1. Yu H, Gan A, Zhang K, et al. [Evaluation of Retrieval-Augmented Generation: A Survey[J]](https://arxiv.org/pdf/2005.11401). arXiv preprint arXiv:2405.07437, 2024.
