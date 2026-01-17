# rag_api.py - FastAPI RAG Service for Remote Access
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import json
from sentence_transformers import SentenceTransformer
from rag_system import LocalPGVectorRAG, OCRConfig

app = FastAPI(title="Financial RAG API", version="1.0.0")

# Global RAG system instance
rag_system = None

class RAGQuery(BaseModel):
    query: str
    limit: int = 5
    similarity_threshold: float = 0.0

class RAGResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    print("Initializing RAG system...")
    
    try:
        # Database configuration
        db_config = {
            'host': 'localhost',
            'database': 'financial_rag',
            'user': 'tanmay',
            'password': '1999',
            'port': 5432
        }
        
        # OCR configuration
        ocr_config = OCRConfig(
            use_ocr=True, 
            ocr_language='eng', 
            dpi=300, 
            use_ocrmypdf=True
        )
        
        # Initialize RAG system
        rag_system = LocalPGVectorRAG(
            db_config, 
            model_name='mukaj/fin-mpnet-base', 
            ocr_config=ocr_config
        )
        
        # Force CPU usage for embedding model
        rag_system.embedding_model = SentenceTransformer('mukaj/fin-mpnet-base', device='cpu')
        
        print("RAG system initialized successfully!")
        
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Financial RAG API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "embedding_model": "mukaj/fin-mpnet-base",
        "database_connected": True  # You could add actual DB connectivity check here
    }

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGQuery):
    """Query the RAG system"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Generate HyDE text and embedding
        hyde_text = rag_system.hyde_generator.generate_hypothetical_document(request.query)
        query_embedding = rag_system.embedding_model.encode(hyde_text)
        
        # Build SQL query
        sql = """
        SELECT content, pdf_name, pdf_link, year, doc_type, chunk_index, ocr_processed,
        1 - (embedding <=> %s::vector) AS similarity
        FROM financial_documents
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY similarity DESC
        LIMIT %s
        """
        
        params = [
            query_embedding.tolist(), 
            query_embedding.tolist(),
            request.similarity_threshold,
            request.limit
        ]
        
        # Execute query
        conn = rag_system.connection_pool.getconn()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        rag_system.connection_pool.putconn(conn)
        
        # Format results
        results = []
        for row in rows:
            content, pdf_name, pdf_link, year, doc_type, chunk_index, ocr_processed, similarity = row
            results.append({
                "content": content,
                "pdf_name": pdf_name,
                "pdf_link": pdf_link,
                "year": year,
                "doc_type": doc_type,
                "chunk_index": chunk_index,
                "ocr_processed": ocr_processed,
                "similarity": float(similarity),
                "preview": content[:300].strip() + "..." if len(content) > 300 else content
            })
        
        return RAGResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_info={
                "original_query": request.query,
                "hyde_query": hyde_text,
                "similarity_threshold": request.similarity_threshold,
                "requested_limit": request.limit
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/query", response_model=RAGResponse)
async def query_rag_get(
    q: str = Query(..., description="Query string"),
    limit: int = Query(5, description="Number of results to return"),
    threshold: float = Query(0.0, description="Minimum similarity threshold")
):
    """GET endpoint for simple queries"""
    request = RAGQuery(query=q, limit=limit, similarity_threshold=threshold)
    return await query_rag(request)

if __name__ == "__main__":
    print("Starting Financial RAG API Server...")
    print("Make sure PostgreSQL with pgvector is running")
    print("API will be available at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")