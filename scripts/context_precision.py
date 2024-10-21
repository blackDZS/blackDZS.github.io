import json
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    api_key="sk-E8zStvbbqI1XAwDKFc880623482549D7B08eEd54D30898F3",
    base_url="https://one-api.s.metames.cn:38443/v1",
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