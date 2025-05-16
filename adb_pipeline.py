"""
title: ArangoDB Graph Entity Streamer Pipeline
author: Neeraj Verma
date: 2025-05-15
version: 1.0
license: MIT
description: A pipeline for streaming relevant entity nodes from an ArangoDB graph using LangChain retriever interface.
requirements: arango, langchain
"""

import os
from typing import List, Union, Generator, Iterator, Any

from arango import ArangoClient
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from schemas import OpenAIChatMessage
from pydantic import Field


class ArangoGraphRetriever(BaseRetriever):
    client: Any = Field()
    graph_name: str = Field()
    start_collection: str = Field(default="GraphNodes")
    edge_collection: str = Field(default="GraphEdges")
    hop: int = Field(default=1)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        aql_find_nodes = f"""
        FOR node IN {self.start_collection}
            FILTER HAS(node, 'text') AND CONTAINS(LOWER(node.text), LOWER("{query.lower()}"))
            RETURN node
        """
        start_nodes = list(self.client.aql.execute(aql_find_nodes))
        docs = [
            Document(page_content=node["text"], metadata={"id": node["_key"]})
            for node in start_nodes
        ]

        for node in start_nodes:
            traversal_query = f"""
            FOR v, e, p IN 1..{self.hop} ANY '{self.start_collection}/{node["_key"]}' {self.edge_collection}
                FILTER HAS(v, 'text')
                RETURN v
            """
            connected_nodes = list(self.client.aql.execute(traversal_query))
            docs.extend([
                Document(page_content=conn["text"], metadata={"id": conn["_key"]})
                for conn in connected_nodes
            ])

        return docs


class Pipeline:
    def __init__(self):
        self.db = None

    async def on_startup(self):
        client = ArangoClient(hosts="http://host.docker.internal:8529")
        self.db = client.db("_system", username="root", password="password")

    async def on_shutdown(self):
        self.db = None

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f">> User message: {user_message}")
        retriever = ArangoGraphRetriever(
            client=self.db,
            graph_name="policies_graph",
            hop=100
        )
        documents = retriever.get_relevant_documents(user_message)

        return "\n---\n".join(doc.page_content for doc in documents)
