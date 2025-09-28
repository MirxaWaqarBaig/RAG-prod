# System Prompt RAG - Swift Car Rental AI Customer Service

An intelligent customer service system based on Retrieval-Augmented Generation (RAG), designed for Swift Car Rental platform, providing Chinese Q&A services.

## 🚀 Project Overview

This project is a complete RAG system that combines:
- **Hybrid Retrieval**: BM25 + Semantic Search
- **Intelligent Chunking**: Dynamic chunking strategy based on document types
- **Dual-Mode Responses**: Concise answers + Detailed analysis
- **Chinese Optimization**: Specialized for Chinese text and business scenarios
- **Containerized Deployment**: Docker + Docker Compose

## 📋 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  ZMQ Broker     │────│  RAG Server     │
│   (Port 8000)   │    │  (Port 5559/60) │    │(GPU Accelerated)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │ (Semantic Cache)│
                       └─────────────────┘
```

## 🛠️ Technology Stack

- **Python 3.10** + PyTorch
- **LlamaIndex** - RAG Framework
- **ChromaDB** - Vector Storage
- **PostgreSQL** - Semantic Cache
- **ZMQ** - Message Queue
- **Docker** - Containerized Deployment
- **GPU Acceleration** - CUDA Support

## 📦 Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone https://github.com/MirxaWaqarBaig/RAG-prod.git
cd RAG-prod

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file:

```bash
# LLM Configuration (choose one)
DEEPSEEK_API_KEY=your_deepseek_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key

# Database Configuration
PG_HOST=localhost
PG_PORT=5432
PG_DBNAME=semantic_cache
PG_USER=postgres
PG_PASSWORD=postgres

# System Configuration
DEVICE=cuda  # or cpu
RAG_SYSTEM_MODE=system_prompt
RAG_RELOAD_INTERVAL=120
SIMILARITY_THRESHOLD=0.75
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

### 3. Prepare Knowledge Base Documents

Place documents in `input/` directory:

```bash
# Example document structure
input/
├── nkb.txt          # Basic Q&A
├── new.txt          # Detailed FAQ
└── other_docs.txt   # Other documents
```

### 4. Download Cache Folder

Download the pre-built cache folder to avoid long model download times:

```bash
# Download the .cache folder from Google Drive
# Link: https://drive.google.com/file/d/1U6EnatdRLHZ4PT0VnNVdGp-asY_Q7BkU/view?usp=sharing

# Extract the downloaded file to the project root
# This will create the .cache/ directory with pre-downloaded models
```

### 5. Build Vector Index

```bash
# Generate chunks and build ChromaDB index
python build_chroma_only.py
```

### 6. Start Services

#### Method 1: Direct Run
```bash
# Start RAG server
python system_rag_server.py serve
```

#### Method 2: Docker Deployment
```bash
# Ensure .cache folder is present (download from Google Drive if needed)
# Link: https://drive.google.com/file/d/1U6EnatdRLHZ4PT0VnNVdGp-asY_Q7BkU/view?usp=sharing

# Build image
docker build -t system-rag:latest .

# Start complete service stack
docker-compose up -d
```

## 🎯 Usage

### API Calls

```bash
# Test script (local development)
chmod +x test_rag.sh
./test_rag.sh

# For production testing, modify test_rag.sh to use:
# http://chatbot.sharestyleai.com:8000/api/text-to-text
```

### Manual API Calls

#### Local Development
```bash
# Normal response
curl -X POST http://localhost:8000/api/text-to-text \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How do I return a car?", "detailed": false}'

# Detailed response
curl -X POST http://localhost:8000/api/text-to-text \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How do I return a car?", "detailed": true}'
```

#### Production Environment
```bash
# Normal response
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How do I return a car?", "detailed": false}'

# Detailed response
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How do I return a car?", "detailed": true}'
```

### Admin Commands

#### Local Development
```bash
# Reload index
curl -X POST http://localhost:8000/api/text-to-text \
     -d '{"prompt": "!reload"}'

# View statistics
curl -X POST http://localhost:8000/api/text-to-text \
     -d '{"prompt": "!stats"}'

# Clear cache
curl -X POST http://localhost:8000/api/text-to-text \
     -d '{"prompt": "!clear"}'

# Get detailed response
curl -X POST http://localhost:8000/api/text-to-text \
     -d '{"prompt": "!detailed"}'
```

#### Production Environment
```bash
# Reload index
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -d '{"prompt": "!reload"}'

# View statistics
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -d '{"prompt": "!stats"}'

# Clear cache
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -d '{"prompt": "!clear"}'

# Get detailed response
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -d '{"prompt": "!detailed"}'
```

## 🔧 Core Features

### 1. Hybrid Retrieval System

- **Semantic Search**: Vector similarity based on Chinese embeddings
- **BM25 Search**: Traditional keyword matching
- **Weight Configuration**: 60% semantic + 40% keywords
- **Result Count**: 6 most relevant document chunks per query

### 2. Intelligent Document Chunking

System automatically detects document types and applies appropriate chunking strategies:

- **Q&A Documents**: Chunking based on Q&A patterns
- **Structured Documents**: Chunking based on header hierarchy
- **Paragraph Documents**: Chunking based on paragraphs
- **Mixed Documents**: Comprehensive chunking strategy

**Chunking Parameters**:
- Size Range: 800-1200 characters
- Overlap Size: 250 characters
- Context Preservation: Maintains document structure

### 3. Dual-Mode Response

#### Normal Response (NORMAL)
- **Length**: 120-180 words
- **Style**: Direct, practical, conclusion-oriented
- **Structure**: Key points + 1-2 bullet lists
- **Content**: Core "what/why/how"

#### Detailed Response (DETAILED)
- **Length**: 350+ words
- **Style**: Comprehensive, multi-paragraph analysis
- **Structure**: Hierarchical titles + examples + processes
- **Content**: Background + steps + precautions + comparisons

### 4. Semantic Caching System

- **PostgreSQL Cache**: Similarity-based response caching
- **Cache Variants**: Normal and detailed responses cached separately
- **Automatic Management**: Cache cleanup and optimization
- **Performance Boost**: Reduces redundant computations

### 5. Auto-Reload

- **File Monitoring**: Monitors `input/` directory changes
- **Reload Interval**: 120 seconds (configurable)
- **Hot Reload**: Updates index without service restart

## 📁 Project Structure

```
system_prompt_rag/
├── README.md                    # Project documentation
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image build
├── docker-compose.yml           # Service orchestration
├── system_rag_server.py         # ZMQ service entry point
├── system_rag_service.py        # Core RAG service
├── core.py                      # LLM and embedding management
├── core_hybrid.py               # Hybrid retrieval implementation
├── optimized_chunking.py         # Intelligent chunking strategy
├── cache.py                     # Semantic cache management
├── build_chroma_only.py         # Index building script
├── test_rag.sh                  # Test script
├── rag_system_prompt.md         # System prompt
├── input/                       # Knowledge base documents
│   ├── nkb.txt
│   └── new.txt
├── input_chunked/               # Chunked documents
├── chromadb/                    # Vector database
├── .cache/                      # Model cache
└── misc/                        # Other files
```

## 🐳 Docker Deployment

### Build Image

```bash
# Build RAG service image
docker build -t system-rag:latest .
```

### Start Service Stack

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f system-rag-server
```

### Service Ports

- **API Gateway**: `http://localhost:8000`
- **PostgreSQL**: `localhost:5433`
- **ZMQ Broker**: `localhost:5559/5560`

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Computing device (cuda/cpu) |
| `RAG_SYSTEM_MODE` | `system_prompt` | RAG mode |
| `RAG_RELOAD_INTERVAL` | `120` | Reload interval (seconds) |
| `SIMILARITY_THRESHOLD` | `0.75` | Similarity threshold |
| `EMBEDDING_MODEL` | `shibing624/text2vec-base-chinese` | Embedding model |

### LLM Configuration

Supports two LLMs:

1. **DeepSeek** (Recommended)
```bash
DEEPSEEK_API_KEY=your_api_key
```

2. **Qwen** (Alibaba Cloud)
```bash
DASHSCOPE_API_KEY=your_api_key
```

## 🔍 Performance Optimization

### GPU Acceleration

```bash
# Ensure NVIDIA drivers and Docker GPU support
docker run --gpus all system-rag:latest
```

### Cache Optimization

- **Semantic Cache**: Reduces redundant computations
- **Model Cache**: Local caching of embedding models
- **Index Cache**: ChromaDB persistent storage

### Memory Management

- **Chunk Size**: 800-1200 characters balances performance and accuracy
- **Batch Processing**: Batch processing of documents and queries
- **Garbage Collection**: Regular cleanup of temporary data

## 🐛 Troubleshooting

### Common Issues

1. **ChromaDB Index Not Found**
```bash
# Rebuild index
python build_chroma_only.py
```

2. **GPU Memory Insufficient**
```bash
# Use CPU mode
export DEVICE=cpu
```

3. **PostgreSQL Connection Failed**
```bash
# Check database configuration
docker-compose logs postgres
```

4. **LLM API Call Failed**
```bash
# Check API key and network connection
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
     https://api.deepseek.com/v1/models
```

### Log Viewing

```bash
# View RAG service logs
docker-compose logs -f system-rag-server

# View all service logs
docker-compose logs -f
```

## 📊 Monitoring and Statistics

### System Statistics

#### Local Development
```bash
# Get system statistics
curl -X POST http://localhost:8000/api/text-to-text \
     -d '{"prompt": "!stats"}'
```

#### Production Environment
```bash
# Get system statistics
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-text \
     -d '{"prompt": "!stats"}'
```

### Performance Metrics

- **Query Encoding Time**: Embedding generation time
- **Document Retrieval Time**: Hybrid search time
- **Cache Hit Rate**: Cache effectiveness statistics
- **Total Response Time**: End-to-end latency

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or suggestions, please:

1. Check the troubleshooting section in this documentation
2. Check the project's Issues page
3. Create a new Issue describing the problem

---

**Note**: This project is specifically designed for Swift Car Rental business scenarios, containing specific business logic and Chinese optimizations. When using in other scenarios, please adjust the system prompts and knowledge base content according to actual needs.
