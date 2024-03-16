# [Agent RAG with Knowledge Graphs](http://ec2-54-80-98-106.compute-1.amazonaws.com:8001/agent/playground/) 🚀

This project is a practical application of the concepts discussed in [Enhancing RAG-based application accuracy by constructing and leveraging knowledge graphs](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)*. It aims to advance the original implementation by integrating its logic into a basic application framework.

## Features

- Incorporation of Knowledge Graphs into Agent RAG for enhanced performance.
- Use of the Tavily search engine as an auxiliary tool.
- Simple deployment via **LangServe**.

## Technical Details

- **Graph Database:** We employ **Neo4J** for our graph database needs.
- **Model:** We utilize two models, `gpt-4-0125-preview` for generative tasks, and `text-embedding-3-small` for embeddings.
- **App Deployment:** The application is hosted on a `t3.xlarge` EC2 instance, ensuring robust performance.
- **App Construction:** Development was carried out using **LangServe**, facilitating a streamlined build process.

## Code Walkthrough

Below are the main components of our application, along with links to their implementation:

| Component                | Link                                                                                                                       |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Application              | [serve.py](https://github.com/jjovalle99/Agent-RAG-Knowledge-Graphs/blob/86f046b92b8d40c020f475f30ce3e98f819c2db4/serve.py)               |
| Step by Step Implementation | [graph_retrieval_playground.ipynb](https://github.com/jjovalle99/Agent-RAG-Knowledge-Graphs/blob/86f046b92b8d40c020f475f30ce3e98f819c2db4/graph_retrieval_playground.ipynb)          |

## Getting Started

Interact with the application via the following URL: [http://ec2-54-80-98-106.compute-1.amazonaws.com:8001/agent/playground/](http://ec2-54-80-98-106.compute-1.amazonaws.com:8001/agent/playground/). To get a feel for its capabilities, try out these queries:
- "What are the Generative AI Practice Requirements?" - This query demonstrates the use of the Enhanced RAG with Knowledge Graphs.
- "Who won the 2001 Copa America?" - This query showcases the integration with TavilySearchResults.
