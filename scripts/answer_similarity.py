import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key="sk-E8zStvbbqI1XAwDKFc880623482549D7B08eEd54D30898F3",
    base_url="https://one-api.s.metames.cn:38443/v1",
    model="text-embedding-3-small",
)

query = "介绍下艾菲尔铁塔"
ground_truth = """
埃菲尔铁塔（法语：Tour Eiffel，/ˈaɪfəl/ [tuʁ‿ɛfɛl] （ⓘ），也常称为巴黎铁塔）是位于法国巴黎第七区、塞纳河畔战神广场的铁制镂空塔，世界著名建筑，也是法国文化象征之一[3]，巴黎城市地标之一，巴黎最高建筑物。正式地址为Rue Anatole-France 5号。

埃菲尔铁塔建成于1889年，初名为“三百米塔”，后得名自其设计师居斯塔夫·埃菲尔。铁塔是世界建筑史上的技术杰作，也是世界上最多人付费参观的名胜古迹，这个为了世界博览会而落成的金属建筑，2011年约有698万人参观[4]，是法国参观人数第二多的文化景点。1986年美国土木工程师协会将该建筑列入国际土木工程历史古迹，1991年，埃菲尔铁塔连同巴黎塞纳河沿岸整座被列入世界遗产。[5]

埃菲尔铁塔以312米的高度，占据世界最高人造建筑的位置长达四十年，直到纽约克莱斯勒大楼的出现，其位于279.11米处的观景平台是欧盟范围内公众能够抵达的最高的观景台，在全欧洲范围内仅次于莫斯科的奥斯坦金诺电视塔。铁塔的总高度曾通过安装天线而多次提高。这些天线曾被用于许多科学实验，现在主要用于发射广播电视信号。
"""
answer = "埃菲尔铁塔（也常称为巴黎铁塔）位于法国巴黎第七区"


def compute_similarity(
    answer, ground_truth, embeddings
) -> float:
    embedding_1 = np.array(embeddings.embed_query(ground_truth))
    embedding_2 = np.array(embeddings.embed_query(answer))
    # Normalization factors of the above embeddings
    norms_1 = np.linalg.norm(embedding_1, keepdims=True)
    norms_2 = np.linalg.norm(embedding_2, keepdims=True)
    embedding_1_normalized = embedding_1 / norms_1
    embedding_2_normalized = embedding_2 / norms_2
    similarity = embedding_1_normalized @ embedding_2_normalized.T
    score = similarity.flatten()
    return score

print("Answer Correctness: ", compute_similarity(answer, ground_truth, embeddings))