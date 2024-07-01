from abc import ABC, abstractmethod
import json
import openai
import os
from dotenv import load_dotenv


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, texts: list[str], filename: str) -> bool:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        return [data.embedding for data in response.data]

    def save(self, texts: list[str], filename: str) -> bool:
        vectors = self.embed(texts)
        data_to_save = [
            {"id": idx, "text": text, "vector": vector}
            for idx, (text, vector) in enumerate(zip(texts, vectors))
        ]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"{filename}に保存されました。")
        return True


if __name__ == "__main__":

    texts = [
        "田中太郎は、大阪生まれの28歳のデザイナーです。趣味は絵を描くこととギター演奏。創造的なプロジェクトに取り組むことが大好きです。",
        "鈴木花子は、福岡生まれの32歳のマーケティングマネージャーです。趣味は料理とランニング。新しいレシピに挑戦することが楽しみです。",
        "高橋次郎は、京都生まれの40歳のデータアナリストです。趣味は読書とチェス。複雑な問題を解決することにやりがいを感じています。",
        "山本美咲は、札幌生まれの27歳の教師です。趣味は映画鑑賞とヨガ。生徒と一緒に成長することが喜びです。",
        "小林健太は、名古屋生まれの30歳のエンジニアです。趣味はサイクリングとボードゲーム。チームでの協力が得意です。",
    ]

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("APIキーがセットされていません。")

    embedder = OpenAIEmbedder(api_key)
    embedder.save(texts, "sample_data.json")
