from usearch.index import Index
import sqlite3
import numpy as np
import os
from functools import lru_cache
from sentence_transformers import CrossEncoder, SentenceTransformer


class AI:
    def __init__(self) -> None:
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # rerank assigns a relevance score of each chunk to the given query
    def rerank(self, query: str, chunks: list[str]):
        # Predict scores (Higher score = More relevant)
        # Cross-encoders take pairs of [query, doc]
        pairs = [[query, chunk] for chunk in chunks]
        return self.reranker.predict(pairs)

    # embed creates vector embeddings of the given sentences in batch
    def embed(self, sentences: list[str]):
        return self.encoder.encode(sentences)


SCHEMA = [
    """
create table memory (
  id integer primary key autoincrement,
  content text not null
)
""",
    """
create table relationship (
  child_memory_id integer not null references memory(id)
  on update cascade
  on delete cascade,
  parent_memory_id integer not null references memory(id)
  on update cascade
  on delete cascade,
  relationship_type text not null,
  primary key (child_memory_id, parent_memory_id)
)
""",
]


@lru_cache(maxsize=32)
def sql_placeholders(args: int, n: int) -> str:
    questions = ("?, " * args)[:-2]
    return (f"({questions}), " * n)[:-2]


class Store:
    ai: AI
    db: sqlite3.Connection
    index: Index

    def __init__(self, ai: AI, prefix="rag") -> None:
        self.ai = ai

        db_path = f"{prefix}-meta.db"
        init_sqlite = not os.path.exists(db_path)
        self.db = sqlite3.connect(db_path)
        if init_sqlite:
            for statement in SCHEMA:
                self.db.execute(statement)

        ndim = self.ai.embed(["This is a test sentence."]).shape[1]
        self.index = Index(
            ndim=ndim,  # Define the number of dimensions in input vectors
            metric="cos",  # Choose 'l2sq', 'ip', 'haversine' or other metric, default = 'cos'
            dtype="bf16",  # Store as 'f64', 'f32', 'f16', 'i8', 'b1'..., default = None
            connectivity=16,  # Optional: Limit number of neighbors per graph node
            expansion_add=128,  # Optional: Control the recall of indexing
            expansion_search=64,  # Optional: Control the quality of the search
            multi=False,  # Optional: Allow multiple vectors per key, default = False
        )
        self.index_path = f"{prefix}-usearch.db"
        if os.path.isfile(self.index_path):
            self.index.load(self.index_path)

    # close closes the RAGStore and all the databases/connections it has open
    def close(self):
        self.db.commit()
        self.db.close()
        self.index.save(self.index_path)

    # add creates a new memory
    def add(self, memories: list[str]) -> list[int]:
        placeholders = sql_placeholders(1, len(memories))
        cursor = self.db.cursor()
        cursor.execute(
            f"insert into memory (content) values {placeholders} returning id",
            tuple(memories),
        )
        new_ids_ints: list[int] = [row[0] for row in cursor.fetchall()]
        cursor.close()
        new_ids = np.array(new_ids_ints)

        embeddings = self.ai.embed(memories)
        self.index.add(new_ids, embeddings)
        return new_ids_ints

    # relate creates a relationship between memories
    def relate(self, child_memory: int, parent_memory: int, type: str):
        self.db.execute(
            "insert into relationship (child_memory_id, parent_memory_id, relationship_type) values (?, ?, ?)",
            (child_memory, parent_memory, type),
        )

    # info returns parent and children memories for a given memory
    def info(self, memory: int):
        # get content
        cursor = self.db.cursor()
        cursor.execute("select content from memory where memory.id = ?", (memory,))
        memory_content = cursor.fetchone()[0]
        cursor.close()

        # get parents
        cursor = self.db.cursor()
        cursor.execute(
            """
select
    r.parent_memory_id,
    r.relationship_type,
    parent.content
from relationship r
inner join memory parent
    on parent.id = r.parent_memory_id
where r.child_memory_id = ?
""",
            (memory,),
        )
        parent_memory_ids: list[int] = []
        parent_relationship_types: list[str] = []
        parent_content: list[str] = []
        for id, rel, memory_content in cursor.fetchall():
            parent_memory_ids.append(id)
            parent_relationship_types.append(rel)
            parent_content.append(memory_content)
        cursor.close()
        parents = {
            "ids": parent_memory_ids,
            "relationships": parent_relationship_types,
            "content": parent_content,
        }

        # get children
        cursor = self.db.cursor()
        cursor.execute(
            """
select
    r.child_memory_id,
    r.relationship_type,
    child.content
from relationship r
inner join memory child
    on child.id = r.child_memory_id
where r.parent_memory_id = ?
""",
            (memory,),
        )
        child_memory_ids: list[int] = []
        child_relationship_types: list[str] = []
        child_content: list[str] = []
        for id, rel, memory_content in cursor.fetchall():
            child_memory_ids.append(id)
            child_relationship_types.append(rel)
            child_content.append(memory_content)
        cursor.close()

        children = {
            "ids": child_memory_ids,
            "relationships": child_relationship_types,
            "content": child_content,
        }

        return {
            "memory_content": memory_content,
            "parents": parents,
            "children": children,
        }

    # rag executes a search for memories
    def rag(self, query: str) -> dict:
        query_embed = self.ai.embed([query])[0]
        matches = self.index.search(query_embed, 256)
        match_ids = [int(match.key) for match in matches]

        placeholders = sql_placeholders(len(matches), 1)
        cursor = self.db.cursor()
        cursor.execute(
            f"select id, content from memory where id in {placeholders}",
            tuple(match_ids),
        )

        ids: list[int] = []
        contents: list[str] = []
        for id, content in cursor.fetchall():
            ids.append(id)
            contents.append(content)
        cursor.close()

        scores: list[float] = [
            float(score) for score in self.ai.rerank(query, contents)
        ]

        return {
            "ids": ids,
            "contents": contents,
            "scores": scores,
        }
