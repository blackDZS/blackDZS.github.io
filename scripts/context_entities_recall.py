import json
from pprint import pprint
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
            Given a text, extract unique entities without repetition. Ensure you consider different forms or mentions of the same entity as a single entity.

            The output should be a well-formatted JSON instance that conforms to the JSON schema below.

            As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
            the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

            Here is the output JSON schema:

            {OutputSchema}
                                                
            Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).


            Examples:

            text: "The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally. Millions of visitors are attracted to it each year for its breathtaking views of the city. Completed in 1889, it was constructed in time for the 1889 World\'s Fair."
            output: 
            {{"entities": ["Eiffel Tower", "Paris", "France", "1889", "World\'s Fair"]}}


            text: "The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement. Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80. It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles."
            output:
            {{"entities": ["Colosseum", "Rome", "Flavian Amphitheatre", "Vespasian", "AD 70", "Titus", "AD 80"]}}

            text: "The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture.Built to protect against invasions from the north, its construction started as early as the 7th century BC. Today, it is a UNESCO World Heritage Site and a major tourist attraction."
            output: 
            {{"entities": ["Great Wall of China", "21,196 kilometers", "7th century BC", "UNESCO World Heritage Site"]}}

            text: "The Apollo 11 mission, which launched on July 16, 1969, marked the first time humans landed on the Moon.Astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins made history, with Armstrong being the first man to step on the lunar surface.         This event was a significant milestone in space exploration."
            output: 
            {{"entities": ["Apollo 11 mission", "July 16, 1969", "Moon", "Neil Armstrong", "Buzz Aldrin", "Michael Collins"]}}

            Your actual task:
            """
        ),
        (
            "human",
            """
            Your actual task:
            text: "{Text}"
            output:
            """
        )
    ]
)    

def get_v(text):
    output_schema = """
    {"type": "object", "properties": {"entities": {"title": "Entities", "type": "array", "items": {"type": "string"}}}, "required": ["entities"]}
    """
    chain = PROMPT | llm
    result = chain.invoke(
        {
            "OutputSchema": output_schema, "Text": text
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
result_context = get_v(".".join(contexts))
pprint(result_context)
result_answer = get_v(answer)
pprint(result_answer)

def compute_score(
    ground_truth_entities, context_entities
) -> float:
    num_entities_in_both = len(
        set(context_entities).intersection(set(ground_truth_entities))
    )
    return num_entities_in_both / (len(ground_truth_entities) + 1e-8)

# for res in result:
#     classes.append(res["attributed"])
print("Context Entites Recall: ", compute_score(result_answer["entities"], result_context["entities"]))