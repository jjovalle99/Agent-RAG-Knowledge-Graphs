import asyncio
from typing import Any, List

from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever, Document


class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ..., description="All the person, organization, or business entities that appear in the text"
    )


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_retriever(question: str, entity_chain, graph) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response])
    return result


class FinalRetriever(BaseRetriever):
    structured_retriever: Any
    vector_index: Any
    entity_chain: Any
    graph: Any

    def get_relevant_documents(self, query: str) -> str:
        # Graph retrieval
        structured_output = self.structured_retriever(
            question=query,
            entity_chain=self.entity_chain,
            graph=self.graph,
        )
        structured_results = [Document(page_content=structured_output)]
        # Vector + Keyword retrieval
        unstructured_results = self.vector_index.similarity_search(query)
        return structured_results + unstructured_results

    async def aget_relevant_documents(self, query: str) -> str:
        # Graph retrieval
        structured_output = await asyncio.to_thread(
            self.structured_retriever,
            question=query,
            entity_chain=self.entity_chain,
            graph=self.graph,
        )
        structured_results = [Document(page_content=structured_output)]

        # Vector + Keyword retrieval
        unstructured_results = await asyncio.to_thread(
            self.vector_index.similarity_search,
            query,
        )

        return structured_results + unstructured_results
