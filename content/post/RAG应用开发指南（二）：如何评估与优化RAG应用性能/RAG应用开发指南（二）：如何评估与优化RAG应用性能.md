---
title: "RAG应用开发指南（二）：如何评估与优化RAG应用性能"
description: 
date: 2024-09-29T17:17:35+08:00
image: 
categories: ["RAG应用开发"]
tags: ["langchain", "RAG应用开发"]
math: true
license: 
hidden: false
comments: true
draft: false
---

## 介绍
在[深入解析RAG原理与实现](/p/rag应用开发指南一深入解析rag原理与实现/ "深入解析RAG原理与实现")一文中，我们探讨了如何构建RAG应用。然而，在成功构建RAG应用后，新的挑战随之而来：如何评估其性能？我们需要什么样的量化指标来进行有效的评估？

## RAG评估

### 评估指标
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

{{< figure src="/images/RAG Metrics.png" width="90%" align="center" title="图 1. RAG 评估指标" >}}

### 计算方法
尽管大型语言模型（LLM）的上下文长度已显著增加，能够将整个文档甚至文本语料库纳入上下文窗口，但在实际应用中，这种做法往往效率低下，容易分散模型的注意力，同时也会增加推理延迟和成本。对于任何特定查询，仅有少量的文本可能具有相关性，但在每次推理时，上下文窗口中的所有token都需被处理。理想情况下，LLM 应该只处理与查询相关的token。因此，在检索增强生成（RAG）应用中，检索过程的主要目标便是精准识别并提取与给定查询相关的token。

#### Context Precison

Context Precision（上下文精度）是信息检索和问答系统中用来评估检索结果质量的重要指标之一。在检索过程中，系统可能会返回多个与用户查询相关的文档，这些文档的内容可能会对生成答案产生不同程度的影响。Context Precision的核心在于衡量在检索到的所有相关文档中，有多少文档实际上对生成用户需要的答案是有帮助的。

\[
   Context Precision = \frac{有用的文档数量}{相关文档数量}
\]


- 相关文档数量 ：指检索系统返回的所有相关文档的总数。
- 有用文档数量 ：指在这些相关文档中，能够为生成正确答案提供有效信息的文档数量。

那么如何衡量文档对于答案生成是否有用呢？传统的NLP方法中可以使用EM算法，或者计算文档和答案之间词级匹配，这些方法仅计算词级别的重叠，忽略了语义上的近似表达。例如，“机器学习用于预测”与“模型被训练来预测”语义相近，但词汇不完全匹配，导致得分偏低。对于开放性问答任务，上下文和答案可能有多种合理表述，传统的机器学习方法无法处理这种问题。同时如果文档包含大量与问题无关的内容，只是偶尔提到了一些相关词汇，会导致误判。为了解决这些问题，可以使用基于大模型的计算方法，依赖于大模型的上下文学习以及推理能力进行判断文档对于答案生成是否有用，如：

```
问题：什么是机器学习？ 
文档：机器学习是一种通过数据训练模型的技术，用于模式识别和预测。
提示：这个文档是否足以回答问题？请在1-5之间打分，并解释原因。
```

下面是对于衡量文档对于答案生成是否有用的Prompt示例，其中包含指令，格式化输出，示例和输入，通过这种设计方式可以让大模型生成格式化的评估结果。

{{< figure src="/images/Context Precision.png" width="90%" align="center" title="图 2. Context Precision Prompt" >}}

因此Context Precision的计算过程如图3所示，首先是根据用户问题检索相关文档，对于每个文档使用图2中的Prompt输入到大模型中，获取大模型对于文档是否有用的判断，最终根据有用的文档数量计算`Context Precision`

{{< figure src="/images/Context Precision Caculate.png" width="90%" align="center" title="图 3. Context Precision 计算过程" >}}

```python
import json
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    api_key="your_api_key",
    model="gpt-4o-mini"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.

            The output should be a well-formatted JSON instance that conforms to the JSON schema below.

            As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
            the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

            Here is the output JSON schema:

            {OutputSchema}
                                                
            Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).


            Examples:

            question: "What can you tell me about Albert Einstein?"
            context: "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius."
            answer: "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics."
            verification:
            {{
                "reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                "verdict": 1
            }}

            question: "who won 2020 icc world cup?"
            context: "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title."
            answer: "England"
            verification:
            {{
                "reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                "verdict": 1
            }}

            question: "What is the tallest mountain in the world?"
            context: "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest."
            answer: "Mount Everest."
            verification:
            {{
                "reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                "verdict": 0
            }}


            Your actual task:
            """
        ),
        (
            "human",
            """
            Your actual task:
            question: "{Query}"
            context: "{Context}"
            answer: "{Answer}"
            verification:
            """
        )
    ]
)    

def get_v(query, context, answer):
    output_schema = """{"description": "Answer for the verification task wether the context was useful.", "type": "object", "properties": {"reason": {"title": "Reason", "description": "Reason for verification", "type": "string"}, "verdict": {"title": "Verdict", "description": "Binary (0/1) verdict of verification", "type": "integer"}}, "required": ["reason", "verdict"]}"""
    chain = PROMPT | llm
    result = chain.invoke(
        {
            "OutputSchema": output_schema, "Query": query, "Context": context, "Answer": answer
        }
    )
    content = result.content
    if content.startswith("```json"):
        content = content.replace("```json", "")
    if content.endswith("```"):
        content = content.replace("```", "")
    try:
        content = json.loads(content)
        verdict = content["verdict"]
    except:
        verdict = None
    print(content)
    return verdict

verdicts = []
query = "艾菲尔铁塔在哪里"
contexts = [
    "埃菲尔铁塔（也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一[3]，巴黎城市地标之一，巴黎最高建筑物",
    "埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。"
]
answer = "艾菲尔铁塔位于巴黎"
for context in contexts:
    verdicts.append(get_v(query=query, context=context, answer=answer))
print("Context Precision: ", sum(verdicts) / len(verdicts))
```

```text
{'reason': '提供的上下文直接描述了艾菲尔铁塔的位置，明确说明它位于法国巴黎，因此上下文对于得出答案是非常有用的。', 'verdict': 1}
{'reason': '上下文提到艾菲尔铁塔的建设和设计师，但没有直接说明它的位置，因此在回答中提到艾菲尔铁塔位于巴黎的内容未得到上下文的支持。', 'verdict': 0}
Context Precision:  0.5
```

#### Context Recall

Context Recall（上下文召回率） 是一种用于评估信息检索系统中检索结果覆盖程度的重要指标。它反映了系统在上下文中检索到的内容与预期信息（即 Ground Truth）之间的匹配程度。在开放问答系统中，Ground Truth 表示预期答案的完整内容或参考答案。Context Recall 的计算流程是通过将 Ground Truth 分解为多个独立的观点（statement），并判断这些句子是否能在检索到的上下文中找到对应内容。

Context Recall 的计算公式如下：

\[
    Context Recall = \frac{上下文中存在的正确观点数量}{Ground Truth 的所有观点数量}
\]

- 上下文中存在的正确观点数量：在给定上下文中能够找到支持的观点的数量。
- Ground Truth 的所有观点数量：答案中涉及的所有观点的数量。

**Prompt 示例**

{{< figure src="/images/Context Recall.png" width="90%" align="center" title="图 4. Context Recall Prompt" >}}


**Context Recall 计算流程**
```
1. 检索上下文：从信息源中获取相关上下文。
2. 大模型判断：将Query, Context, Ground Truth 填入 Prompt 交给大模型，判断是否在上下文中找到匹配的观点(statement)。
3. 计算 Context Recall：根据分类为1的statement和总statement计算召回率。
```

下图展示了 Context Recall 的计算过程：

{{< figure src="/images/Context Recall Caculate.png" width="90%" align="center" title="图 5. Context Recall 计算过程" >}}

```python
import json
from pprint import pprint
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    api_key="your_api_key",
    model="gpt-4o-mini"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason.

            The output should be a well-formatted JSON instance that conforms to the JSON schema below.

            As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
            the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

            Here is the output JSON schema:

            {OutputSchema}
                                                
            Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).


            Examples:

            question: "What can you tell me about albert Albert Einstein?"
            context: "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called \'the world\'s most famous equation\'. He received the 1921 Nobel Prize in Physics \'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect\', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius."
            answer: "Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895"
            classification: 
            [{{"statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.", "attributed": 1, "reason": "The date of birth of Einstein is mentioned clearly in the context."}}, 
            {{"statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.", "attributed": 1, "reason": "The exact sentence is present in the given context."}}, 
            {{"statement": "He published 4 papers in 1905.", "attributed": 0, "reason": "There is no mention about papers he wrote in the given context."}}, 
            {{"statement": "Einstein moved to Switzerland in 1895.", "attributed": 0, "reason": "There is no supporting evidence for this in the given context."}}]

            question: "who won 2020 icc world cup?"
            context: "The 2022 ICC Men\'s T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men\'s T20 World Cup title."\nanswer: "England"
            classification: 
            [{{"statement": "England won the 2022 ICC Men\'s T20 World Cup.", "attributed": 1, "reason": "From context it is clear that England defeated Pakistan to win the World Cup."}}]

            question: "What is the primary fuel for the Sun?"
            context: "NULL"
            answer: "Hydrogen"
            classification: 
            [{{"statement": "The Sun\'s primary fuel is hydrogen.", "attributed": 0, "reason": "The context contains no information"}}]

            Your actual task:
            """
        ),
        (
            "human",
            """
            Your actual task:
            question: "{Query}"
            context: "{Context}"
            answer: "{Answer}"
            classification:
            """
        )
    ]
)    

def get_v(query, context, answer):
    output_schema = """
    {"type": "array", "items": {"$ref": "#/definitions/ContextRecallClassificationAnswer"}, "definitions": {"ContextRecallClassificationAnswer": {"title": "ContextRecallClassificationAnswer", "type": "object", "properties": {"statement": {"title": "Statement", "type": "string"}, "attributed": {"title": "Attributed", "type": "integer"}, "reason": {"title": "Reason", "type": "string"}}, "required": ["statement", "attributed", "reason"]}}}
    """
    chain = PROMPT | llm
    result = chain.invoke(
        {
            "OutputSchema": output_schema, "Query": query, "Context": context, "Answer": answer
        }
    )
    content = result.content
    if content.startswith("```json"):
        content = content.replace("```json", "")
    if content.endswith("```"):
        content = content.replace("```", "")
    try:
        content = json.loads(content)
    except:
        # verdict = None
        pass
    return content

classes = []
query = "艾菲尔铁塔在哪里"
contexts = [
    "埃菲尔铁塔（也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一[3]，巴黎城市地标之一，巴黎最高建筑物",
    "埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。"
]
answer = """
埃菲尔铁塔（法语：Tour Eiffel，/ˈaɪfəl/ [tuʁ‿ɛfɛl] （ⓘ），也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一[3]，巴黎城市地标之一，巴黎最高建筑物。正式地址为Rue Anatole-France 5号。

埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。铁塔是世界建筑史上的技术杰作，也是世界上最多人付费参观的名胜古迹，这个为了世界博览会而落成的金属建筑，2011年约有698万人参观[4]，是法国参观人数第二多的文化景点。1986年美国土木工程师协会将该建筑列入国际土木工程历史古迹，1991年，埃菲尔铁塔连同巴黎塞纳河沿岸整座被列入世界遗产。[5]

埃菲尔铁塔以312米的高度，占据世界最高人造建筑的位置长达四十年，直到纽约克莱斯勒大楼的出现，其位于279.11米处的观景平台是欧盟范围内公众能够抵达的最高的观景台，在全欧洲范围内仅次于莫斯科的奥斯坦金诺电视塔。铁塔的总高度曾通过安装天线而多次提高。这些天线曾被用于许多科学实验，现在主要用于发射广播电视信号。
"""
result = get_v(query, ".".join(contexts), answer)
pprint(result[:3])
for res in result:
    classes.append(res["attributed"])
print("Context Recall: ", sum(classes) / len(classes))
```

```text
[{'attributed': 1,
  'reason': 'This statement is directly supported by the context, which '
            "describes the Eiffel Tower's location and significance.",
  'statement': '埃菲尔铁塔（法语：Tour Eiffel，/ˈaɪfəl/ [tuʁ‿ɛfɛl] '
               '（ⓘ），也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一，巴黎城市地标之一，巴黎最高建筑物。'},
 {'attributed': 0,
  'reason': 'The context does not mention this specific address.',
  'statement': '正式地址为Rue Anatole-France 5号。'},
 {'attributed': 1,
  'reason': 'This information is directly stated in the context.',
  'statement': '埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。'},
 {'attributed': 0,
  'reason': "While the context discusses the Eiffel Tower's significance, it "
            'does not specifically refer to it as a technological masterpiece '
            'or the most visited paid monument.',
  'statement': '铁塔是世界建筑史上的技术杰作，也是世界上最多人付费参观的名胜古迹，'},
 {'attributed': 0,
  'reason': 'The context does not provide specific visitor statistics or '
            'ranking.',
  'statement': '这个为了世界博览会而落成的金属建筑，2011年约有698万人参观，是法国参观人数第二多的文化景点。'},
 {'attributed': 0,
  'reason': 'The context does not mention the inclusion in engineering history '
            'landmarks or UNESCO World Heritage status.',
  'statement': '1986年美国土木工程师协会将该建筑列入国际土木工程历史古迹，1991年，埃菲尔铁塔连同巴黎塞纳河沿岸整座被列入世界遗产。'},
 {'attributed': 0,
  'reason': 'The context does not provide specific height or duration of being '
            'the tallest man-made structure.',
  'statement': '埃菲尔铁塔以312米的高度，占据世界最高人造建筑的位置长达四十年，'},
 {'attributed': 0,
  'reason': 'The context does not mention the installation of antennas or '
            'height increases.',
  'statement': '铁塔的总高度曾通过安装天线而多次提高。'},
 {'attributed': 0,
  'reason': 'There is no mention of antennas being used for scientific '
            'experiments or broadcasting.',
  'statement': '这些天线曾被用于许多科学实验，现在主要用于发射广播电视信号。'}]
Context Recall:  0.2222222222222222
❯ python scripts/context_recall.py
[{'attributed': 1,
  'reason': '该句描述了埃菲尔铁塔的位置和其重要性，与上下文一致。',
  'statement': '埃菲尔铁塔（法语：Tour Eiffel，/ˈaɪfəl/ [tuʁ‿ɛfɛl] '
               '（ⓘ），也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一，巴黎城市地标之一，巴黎最高建筑物。'},
 {'attributed': 0,
  'reason': '上下文中并未提及埃菲尔铁塔的具体地址。',
  'statement': '正式地址为Rue Anatole-France 5号。'},
 {'attributed': 1,
  'reason': '该句中的信息与上下文一致，明确提到其建成时间和命名。',
  'statement': '埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。'},
 {'attributed': 0,
  'reason': '上下文没有提到关于参观人数或其作为技术杰作的具体描述。',
  'statement': '铁塔是世界建筑史上的技术杰作，也是世界上最多人付费参观的名胜古迹，这个为了世界博览会而落成的金属建筑，2011年约有698万人参观，是法国参观人数第二多的文化景点。'},
 {'attributed': 0,
  'reason': '上下文未提及关于建筑的历史遗产地位的信息。',
  'statement': '1986年美国土木工程师协会将该建筑列入国际土木工程历史古迹，1991年，埃菲尔铁塔连同巴黎塞纳河沿岸整座被列入世界遗产。'},
 {'attributed': 0,
  'reason': '上下文中没有提到埃菲尔铁塔的高度或其历史地位。',
  'statement': '埃菲尔铁塔以312米的高度，占据世界最高人造建筑的位置长达四十年，直到纽约克莱斯勒大楼的出现。'},
 {'attributed': 0,
  'reason': '上下文没有包含关于观景平台的高度或排名的信息。',
  'statement': '其位于279.11米处的观景平台是欧盟范围内公众能够抵达的最高的观景台，在全欧洲范围内仅次于莫斯科的奥斯坦金诺电视塔。'},
 {'attributed': 0,
  'reason': '上下文没有提及天线或科学实验相关的信息。',
  'statement': '铁塔的总高度曾通过安装天线而多次提高。这些天线曾被用于许多科学实验，现在主要用于发射广播电视信号。'}]

Context Recall:  0.25
```

## 引用
1. Yu H, Gan A, Zhang K, et al. [Evaluation of Retrieval-Augmented Generation: A Survey[J]](https://arxiv.org/pdf/2005.11401). arXiv preprint arXiv:2405.07437, 2024.
2. https://research.trychroma.com/evaluating-chunking
3. https://github.com/explodinggradients/ragas
