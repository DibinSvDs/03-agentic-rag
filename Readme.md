# 03 — Agentic RAG

A hands-on project for understanding **Agentic RAG from first principles**, without using LangChain or LangGraph.

The project combines:

* Retrieval-Augmented Generation (RAG)
* LLM tool calling
* Agent decision-making
* ChromaDB vector search
* Local Ollama models
* Grounded answers

The goal is to understand what happens underneath an Agentic RAG application before using higher-level AI frameworks.

---

## What is Agentic RAG?

Traditional RAG follows a relatively fixed workflow:

```text
User
 ↓
Retrieve relevant information
 ↓
LLM
 ↓
Answer
```

An AI agent can decide whether it needs to use a tool:

```text
User
 ↓
LLM
 ↓
Decide whether a tool is needed
 ↓
Tool
 ↓
LLM
 ↓
Answer
```

Agentic RAG combines the two:

```text
User
 ↓
Qwen3
 ↓
Does the question require company information?
       ↙                    ↘
     YES                    NO
      ↓                      ↓
Retrieval Tool          Answer directly
      ↓
   ChromaDB
      ↓
Retrieved documents
      ↓
    Qwen3
      ↓
Final answer
```

The important idea is that **retrieval itself becomes a tool that the agent can choose to use.**

---

## Project Goal

This project was created to understand:

1. How an LLM decides whether to use a tool
2. How native tool calling works
3. How retrieval can be exposed as an agent tool
4. How embeddings are used to search documents
5. How ChromaDB stores and retrieves vectors
6. How retrieved context is passed back to the LLM
7. How to keep answers grounded in retrieved information
8. What happens when the knowledge base does not contain the answer

---

## Technologies

* Python
* Ollama
* Qwen3 4B (`qwen3:4b`)
* `nomic-embed-text`
* ChromaDB
* PyMuPDF
* Native Ollama tool calling

No OpenAI API is required.

No LangChain or LangGraph is used.

---

## Project Structure

```text
03-agentic-rag/
│
├── app.py
├── tools.py
├── ingest.py
│
├── documents/
│   └── company_handbook.txt
│
├── vector_db/
│
├── requirements.txt
└── .gitignore
```

### File responsibilities

**`ingest.py`**

Reads the company handbook, creates meaningful chunks, generates embeddings using `nomic-embed-text`, and stores them in ChromaDB.

**`tools.py`**

Contains the retrieval function:

```python
search_company_documents()
```

It converts the user's search question into an embedding, searches ChromaDB, and returns the most relevant document chunks.

It also contains the tool schema that tells Qwen that the retrieval function is available.

**`app.py`**

Runs the agent loop.

It sends the user question to Qwen, checks whether Qwen requested the retrieval tool, executes the function, sends the result back to Qwen, and finally prints the answer.

**`documents/company_handbook.txt`**

A larger simulated company employee handbook containing multiple policy sections.

**`vector_db/`**

The persistent local ChromaDB database containing document embeddings.

---

## Knowledge Base

The project uses a simulated company employee handbook containing sections such as:

* Leave and time off
* Working hours
* Remote work
* Company laptops and IT equipment
* Software installation
* Password and account security
* Information security
* Data privacy
* Travel and expenses
* Performance reviews
* Promotions
* Training
* Workplace conduct
* Business continuity
* IT support
* Employee benefits
* General policy questions

The handbook is intentionally larger than the simple documents used in the earlier RAG project so that retrieval becomes meaningful.

---

## How Ingestion Works

Run:

```bash
python ingest.py
```

The ingestion pipeline is:

```text
company_handbook.txt
        ↓
Read document
        ↓
Split into meaningful sections
        ↓
Create chunks
        ↓
nomic-embed-text
        ↓
Generate embeddings
        ↓
ChromaDB
```

The current handbook produces approximately 18 chunks.

Each chunk contains a meaningful policy section rather than an isolated sentence.

---

## How Retrieval Works

When the retrieval function receives a question:

```text
User question
      ↓
nomic-embed-text
      ↓
Question embedding
      ↓
ChromaDB similarity search
      ↓
Top relevant chunks
```

For example:

```text
What should employees do if they have a security-related incident?
```

The retrieval system can identify sections such as:

```text
7. INFORMATION SECURITY

14. BUSINESS CONTINUITY AND EMERGENCIES

15. IT SUPPORT
```

Those chunks are then returned to the LLM.

---

## How the Agent Works

The agent uses Qwen3's native tool-calling capability.

The tool is described to Qwen using a schema containing:

```text
search_company_documents
```

Qwen decides whether that tool is useful.

### Company policy question

Example:

```text
What should employees do if they have a security-related incident?
```

The flow becomes:

```text
User
 ↓
Qwen3
 ↓
Tool required
 ↓
search_company_documents()
 ↓
ChromaDB
 ↓
Relevant policy sections
 ↓
Qwen3
 ↓
Final answer
```

The application displays:

```text
[Agent called function: search_company_documents]
```

This makes the agent's tool usage visible during development.

---

## When No Tool Is Needed

The agent does not have to retrieve information for every question.

For example:

```text
What is 2 + 2?
```

Qwen can decide that the company knowledge base is not relevant.

The flow becomes:

```text
User
 ↓
Qwen3
 ↓
No tool needed
 ↓
Answer directly
```

No retrieval function is called.

This demonstrates the difference between:

**RAG**

```text
Always retrieve → answer
```

and:

**Agentic RAG**

```text
LLM decides whether retrieval is necessary
```

---

## Grounded Answers

A major problem with RAG systems is that an LLM may use its general pretrained knowledge when the retrieved documents do not contain enough information.

For example:

```text
What is the company's maternity leave policy in Brazil?
```

The company handbook does not contain a specific Brazil maternity-leave policy.

Without grounding instructions, the model may attempt to answer using general knowledge.

This project adds a system instruction telling Qwen to:

* use the retrieved company documents
* not use general knowledge to fill missing information
* explicitly say when the information is not available

The resulting behavior is:

```text
Retrieved documents
        ↓
Is the answer supported?
       ↙ ↘
     YES  NO
      ↓    ↓
   Answer  Say information
           is unavailable
```

This is an important principle for enterprise RAG systems:

> If the knowledge base does not contain the required information, the system should acknowledge that instead of inventing an answer.

---

## Example

### Question

```text
What should employees do if they have a security-related incident?
```

### Agent behavior

```text
[Agent called function: search_company_documents]
```

The retrieval system returns the Information Security section and related sections.

Qwen then produces a grounded answer based on the retrieved company policies.

---

## Agent vs RAG vs Agentic RAG

### RAG

```text
Question
 ↓
Retrieve
 ↓
Context
 ↓
LLM
 ↓
Answer
```

The retrieval step is part of the predefined pipeline.

### Agent

```text
Question
 ↓
LLM
 ↓
Choose tool
 ↓
Tool
 ↓
LLM
 ↓
Answer
```

The LLM decides which action to take.

### Agentic RAG

```text
Question
 ↓
LLM
 ↓
Choose whether retrieval is needed
 ↓
Retrieval tool
 ↓
Vector database
 ↓
Context
 ↓
LLM
 ↓
Answer
```

The retrieval capability is exposed as a tool that the agent can choose to use.

---

## Why Build This Without LangChain?

Frameworks such as LangChain and LangGraph can make agentic applications easier to build.

However, using them too early can hide the underlying mechanics.

This project intentionally implements the core loop manually:

```python
while True:

    response = ollama.chat(...)

    tool_calls = response["message"].get(
        "tool_calls",
        []
    )

    if not tool_calls:
        break

    execute_tool()

    send_tool_result_back_to_llm()
```

Understanding this loop makes higher-level agent frameworks much easier to understand later.

---

## Key Lessons

### 1. Retrieval can be a tool

An agent does not need a special "RAG mode."

The retrieval function can simply be exposed as a tool.

### 2. The LLM decides when to retrieve

Python does not need to contain rules such as:

```python
if "policy" in question:
    search()
```

Instead, Qwen decides whether the retrieval tool is appropriate.

### 3. Retrieval quality matters

A correct agent decision does not guarantee a correct answer.

If retrieval returns irrelevant chunks:

```text
Correct agent decision
        ↓
Bad retrieval
        ↓
Bad context
        ↓
Potentially bad answer
```

This is why chunking, embeddings, retrieval parameters, reranking, and evaluation matter.

### 4. More retrieved documents are not always better

Retrieving more chunks can introduce irrelevant information.

The goal is not:

```text
Retrieve everything
```

The goal is:

```text
Retrieve the most useful information
```

### 5. Grounding is important

The LLM should not automatically fill missing information with its pretrained knowledge when answering questions about an enterprise knowledge base.

---

## Running the Project

### 1. Activate the virtual environment

Git Bash:

```bash
source .venv/Scripts/activate
```

### 2. Make sure Ollama is running

The required models are:

```bash
ollama list
```

You should have:

```text
qwen3:4b
nomic-embed-text
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the vector database

```bash
python ingest.py
```

### 5. Run the agent

```bash
python app.py
```

---

## Learning Path

This project is the third stage of the AI engineering learning path.

```text
01 — RAG
     ↓
Understand retrieval and vector databases
     ↓
02 — AI Agent
     ↓
Understand LLM tool calling and agent loops
     ↓
03 — Agentic RAG
     ↓
Combine agents + retrieval
     ↓
04 — Agentic AI
     ↓
Build more capable multi-step agents
```

---

## Future Improvements

Possible improvements for this project include:

* Better chunking strategies
* Chunk overlap
* Retrieval score thresholds
* Metadata filtering
* Reranking
* Source citations
* Structured tool outputs
* Retrieval evaluation
* Answer evaluation
* Conversation memory
* Multiple retrieval tools
* Multiple agents
* LangGraph implementation
* Production deployment

---

## Main Takeaway

The most important concept from this project is:

> **Agentic RAG is essentially an agent that has retrieval as one of its available tools.**

Instead of blindly retrieving documents for every question, the LLM can decide:

```text
Do I need company information?
        ↓
      YES → retrieve → use context → answer
        ↓
       NO → answer directly
```

This project demonstrates that workflow using Python, Ollama, Qwen3, ChromaDB, and native tool calling without relying on an agent framework.


