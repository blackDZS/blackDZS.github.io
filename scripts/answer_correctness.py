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
            Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories:

            - TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth,
            - FP (false positive): statements present in the answer but not directly supported by any statement in ground truth,
            - FN (false negative): statements found in the ground truth but not present in answer.

            Each statement can only belong to one of the categories. Provide a reason for each classification.

            The output should be a well-formatted JSON instance that conforms to the JSON schema below.
            As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
            the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

            Here is the output JSON schema:

            {OutputSchema}
                                                
            Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).


            Examples:

            question: "What powers the sun and what is its primary function?"
            answer: 
            ["The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "The primary function of the sun is to provide light to the solar system."]
            ground_truth: 
            ["The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "This fusion process in the sun\'s core releases a tremendous amount of energy.", "The energy from the sun provides heat and light, which are essential for life on Earth.", "The sun\'s light plays a critical role in Earth\'s climate system.", "Sunlight helps to drive the weather and ocean currents."]
            classification: 
            {{"TP": [{{"statement": "The primary function of the sun is to provide light to the solar system.", "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun\'s energy."}}], "FP": [{{"statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."}}], "FN": [{{"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "reason": "This accurate description of the sun’s power source is not included in the answer."}}, {{"statement": "This fusion process in the sun\'s core releases a tremendous amount of energy.", "reason": "This process and its significance are not mentioned in the answer."}}, {{"statement": "The energy from the sun provides heat and light, which are essential for life on Earth.", "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."}}, {{"statement": "The sun\'s light plays a critical role in Earth\'s climate system.", "reason": "This broader impact of the sun’s light on Earth\'s climate system is not addressed in the answer."}}, {{"statement": "Sunlight helps to drive the weather and ocean currents.", "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."}}]}}

            question: "What is the boiling point of water?"
            answer: 
            ["The boiling point of water is 100 degrees Celsius at sea level"]
            ground_truth: 
            ["The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "The boiling point of water can change with altitude."]
            classification: 
            {{"TP": [{{"statement": "The boiling point of water is 100 degrees Celsius at sea level", "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."}}], "FP": [], "FN": [{{"statement": "The boiling point of water can change with altitude.", "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."}}]}}

            Your actual task:
            """
        ),
        (
            "human",
            """
            Your actual task:
            question:"{Query}"
            answer:"{Answer}"
            ground_truth:"{GroundTruth}"
            classification:
            """
        )
    ]
)    

def get_v(query, context, answer, ground_truth):
    output_schema = """
    {"type": "object", "properties": {"TP": {"title": "Tp", "type": "array", "items": {"type": "object"}}, "FP": {"title": "Fp", "type": "array", "items": {"type": "object"}}, "FN": {"title": "Fn", "type": "array", "items": {"type": "object"}}}, "required": ["TP", "FP", "FN"]}
    """
    chain = PROMPT | llm
    result = chain.invoke(
        {
            "OutputSchema": output_schema, "Query": query, "Context": context, "Answer": answer, "GroundTruth": ground_truth
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
query = "介绍下艾菲尔铁塔"
contexts = [
    "埃菲尔铁塔（也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一[3]，巴黎城市地标之一，巴黎最高建筑物",
    "埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。"
]
ground_truth = """
埃菲尔铁塔（法语：Tour Eiffel，/ˈaɪfəl/ [tuʁ‿ɛfɛl] （ⓘ），也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一[3]，巴黎城市地标之一，巴黎最高建筑物。正式地址为Rue Anatole-France 5号。

埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。铁塔是世界建筑史上的技术杰作，也是世界上最多人付费参观的名胜古迹，这个为了世界博览会而落成的金属建筑，2011年约有698万人参观[4]，是法国参观人数第二多的文化景点。1986年美国土木工程师协会将该建筑列入国际土木工程历史古迹，1991年，埃菲尔铁塔连同巴黎塞纳河沿岸整座被列入世界遗产。[5]

埃菲尔铁塔以312米的高度，占据世界最高人造建筑的位置长达四十年，直到纽约克莱斯勒大楼的出现，其位于279.11米处的观景平台是欧盟范围内公众能够抵达的最高的观景台，在全欧洲范围内仅次于莫斯科的奥斯坦金诺电视塔。铁塔的总高度曾通过安装天线而多次提高。这些天线曾被用于许多科学实验，现在主要用于发射广播电视信号。
"""
answer = "埃菲尔铁塔（也常称为巴黎铁塔）位于法国巴黎第七区"
result = get_v(query, ".".join(contexts), answer, ground_truth)
pprint(result)


def compute_statement_presence(
    prediction
) -> float:
    tp = len(prediction["TP"])
    fp = len(prediction["FP"])
    fn = len(prediction["FN"])
    score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
    return score

print("Answer Correctness: ", compute_statement_presence(result))