# RAG System Query Flow Diagram

```mermaid
graph TD
    %% Frontend Layer
    A[User Types Query] --> B["script.js: sendMessage()"]
    B --> C["HTTP POST /api/query<br/>- query<br/>- session_id"]
    
    %% API Layer
    C --> D["app.py: query_documents()"]
    D --> E["Create/Get Session ID"]
    E --> F["RAGSystem.query()"]
    
    %% RAG System Layer
    F --> G["rag_system.py"]
    G --> H["Get Conversation History<br/>SessionManager"]
    G --> I["Create AI Prompt<br/>Answer this question about course materials"]
    I --> J["AIGenerator.generate_response()"]
    
    %% AI Layer
    J --> K["ai_generator.py<br/>Send to Claude API"]
    K --> L{Claude Decision}
    L -->|Direct Answer| M[Return Response]
    L -->|Use Tools| N[Tool Execution]
    
    %% Tool Execution Branch
    N --> O["search_tools.py<br/>CourseSearchTool.execute()"]
    O --> P["vector_store.py<br/>VectorStore.search()"]
    P --> Q[ChromaDB<br/>Semantic Search]
    Q --> R["Search Results<br/>+ Metadata"]
    R --> S[Format Results<br/>Track Sources]
    S --> T[Send Results to Claude]
    T --> U[Claude Final Response<br/>Using Search Context]
    
    %% Response Chain
    U --> V["RAGSystem Returns<br/>response + sources"]
    M --> V
    V --> W[Update Session History]
    W --> X["FastAPI Returns<br/>QueryResponse JSON"]
    X --> Y["Frontend Receives<br/>- answer<br/>- sources<br/>- session_id"]
    Y --> Z["Display Answer<br/>+ Collapsible Sources"]
    
    %% Storage Components
    subgraph "Storage Layer"
        Q1[ChromaDB Collections]
        Q2[Course Metadata]
        Q3[Course Content Chunks]
        Q4[Vector Embeddings]
    end
    
    Q --> Q1
    Q1 --> Q2
    Q1 --> Q3
    Q1 --> Q4
    
    %% Session Management
    subgraph "Session Management"
        H1[SessionManager]
        H2[Conversation History]
        H3[Session Storage]
    end
    
    H --> H1
    H1 --> H2
    H1 --> H3
    W --> H1
    
    %% Styling
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef rag fill:#e8f5e8
    classDef ai fill:#fff3e0
    classDef tools fill:#fce4ec
    classDef storage fill:#f1f8e9
    
    class A,B,C,Y,Z frontend
    class D,E,X api
    class F,G,H,I,V,W rag
    class J,K,L,M,U ai
    class N,O,P,Q,R,S,T tools
    class Q1,Q2,Q3,Q4,H1,H2,H3 storage
```

## Key Components:

### 1. **Frontend Flow** (Blue)
- User interaction → HTTP request → Response display

### 2. **API Layer** (Purple) 
- FastAPI endpoint handling → Session management

### 3. **RAG Orchestration** (Green)
- Query processing → History management → Response coordination

### 4. **AI Processing** (Orange)
- Claude API interaction → Tool decision making

### 5. **Tool Execution** (Pink)
- Semantic search → Vector retrieval → Result formatting

### 6. **Storage Layer** (Light Green)
- ChromaDB collections → Course data → Embeddings

## Decision Points:
- **Claude Decision**: Direct answer vs tool usage
- **Tool Results**: Search success → Format → Send to Claude
- **Session Management**: History retrieval and updates

## Data Flow:
1. **Query**: User input → API → RAG → AI
2. **Tool Usage**: AI → Search → Vector DB → Results → AI
3. **Response**: AI → RAG → API → Frontend
4. **Sources**: Tracked through tool execution → UI display