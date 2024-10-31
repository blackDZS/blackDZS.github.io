import json
from pprint import pprint
from langchain_openai import ChatOpenAI
from pysbd import Segmenter
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    api_key="sk-E8zStvbbqI1XAwDKFc880623482549D7B08eEd54D30898F3",
    base_url="https://one-api.s.metames.cn:38443/v1",
    model="gpt-4o-mini"
)

STATEMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.

            The output should be a well-formatted JSON instance that conforms to the JSON schema below.
            As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
            the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

            Here is the output JSON schema:

            {OutputSchema}
                                                
            Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).


            Examples:
            question: "Who was Albert Einstein and what is he best known for?"
            answer: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."
            sentences: 
            "0:He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. 
            1:He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."
            analysis:
            [{{"sentence_index": 0, "simpler_statements": ["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time."]}}, {{"sentence_index": 1, "simpler_statements": ["Albert Einstein was best known for developing the theory of relativity.", "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}}]

            Your actual task:
            """
        ),
        (
            "human",
            """
            Your actual task:
            question:"{Query}"
            answer:"{Answer}"
            sentences:"{Sentences}"
            analysis:
            """
        )
    ]
)    

def get_statements(query, answer, sentences):
    output_schema = """
    {"type": "array", "items": {"$ref": "#/definitions/Statements"}, "definitions": {"Statements": {"title": "Statements", "type": "object", "properties": {"sentence_index": {"title": "Sentence Index", "description": "Index of the sentence from the statement list", "type": "integer"}, "simpler_statements": {"title": "Simpler Statements", "description": "the simpler statements", "type": "array", "items": {"type": "string"}}}, "required": ["sentence_index", "simpler_statements"]}}}
    """
    chain = STATEMENTS_PROMPT | llm
    result = chain.invoke(
        {
            "OutputSchema": output_schema, "Query": query, "Answer": answer, "Sentences": sentences
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

seg = Segmenter(language="zh")
statements = seg.segment(answer)

result = get_statements(query, answer, statements)
pprint(result)


FATHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.
            
            The output should be a well-formatted JSON instance that conforms to the JSON schema below.
            As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
            the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

            Here is the output JSON schema:

            {OutputSchema}
                                                
            Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).


            Examples:
            context: "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects."
            statements: ["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]
            answer: [{{"statement": "John is majoring in Biology.", "reason": "John\'s major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", "verdict": 0}}, {{"statement": "John is taking a course on Artificial Intelligence.", "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.", "verdict": 0}}, {{"statement": "John is a dedicated student.", "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.", "verdict": 1}}, {{"statement": "John has a part-time job.", "reason": "There is no information given in the context about John having a part-time job.", "verdict": 0}}]

            context: "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy."
            statements: ["Albert Einstein was a genius."]
            answer: [{{"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated", "verdict": 0}}]
            
            Your actual task:
            """
        ),
        (
            "human",
            """
            Your actual task:
            context:"{Context}"
            statements:"{Statements}"
            answer:
            """
        )
    ]
)


def get_fathfullness_results(context, statements):
    output_schema = """
    {"type": "array", "items": {"$ref": "#/definitions/StatementFaithfulnessAnswer"}, "definitions": {"StatementFaithfulnessAnswer": {"title": "StatementFaithfulnessAnswer", "type": "object", "properties": {"statement": {"title": "Statement", "description": "the original statement, word-by-word", "type": "string"}, "reason": {"title": "Reason", "description": "the reason of the verdict", "type": "string"}, "verdict": {"title": "Verdict", "description": "the verdict(0/1) of the faithfulness.", "type": "integer"}}, "required": ["statement", "reason", "verdict"]}}}
    """
    chain = FATHFULNESS_PROMPT | llm
    result = chain.invoke(
        {
            "OutputSchema": output_schema, "Context": context, "Statements": statements
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

results = get_fathfullness_results(".".join(contexts), statements)
print(results)

verdicts = [_["verdict"] for _ in results]
print("Answer Fathfulness: ", sum(verdicts) / len(verdicts))