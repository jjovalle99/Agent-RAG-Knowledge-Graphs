import asyncio
from typing import Any, List, Sequence

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    return Relationship(source=source, target=target, type=rel.type.replace(" ", "_").upper())


def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    return Node(id=node.id.title(), type=node.type.capitalize())


class AsyncLLMGraphTransformer(LLMGraphTransformer):
    async def aprocess_response(self, document: Document) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a graph document.
        """
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text})
        nodes = [map_to_base_node(node) for node in raw_schema.nodes] if raw_schema.nodes else []
        relationships = (
            [map_to_base_relationship(rel) for rel in raw_schema.relationships] if raw_schema.relationships else []
        )

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_relationships and self.allowed_nodes:
                nodes = [node for node in nodes if node.type in self.allowed_nodes]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type in self.allowed_relationships
                    and rel.source.type in self.allowed_nodes
                    and rel.target.type in self.allowed_nodes
                ]
            elif self.allowed_nodes and not self.allowed_relationships:
                nodes = [node for node in nodes if node.type in self.allowed_nodes]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type in self.allowed_nodes and rel.target.type in self.allowed_nodes
                ]
            if self.allowed_relationships and not self.allowed_nodes:
                relationships = [rel for rel in relationships if rel.type in self.allowed_relationships]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    async def aconvert_to_graph_documents(self, documents: Sequence[Document]) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = []
        for document in documents:
            task = asyncio.create_task(self.aprocess_response(document))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results
