#!/usr/bin/env python3

"""
Query Script for Local PGVector RAG System (CPU-only, top 5 chunks)
"""

import sys
from sentence_transformers import SentenceTransformer
from rag_system import LocalPGVectorRAG, OCRConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"<your query>\"")
        sys.exit(1)

    query = sys.argv[1]

    db_config = {
        'host': "http://0.0.0.0:8001",
        'database': 'financial_rag',
        'user': 'tanmay',
        'password': '1999',
        'port': 5432
    }

    ocr_config = OCRConfig(use_ocr=True, ocr_language='eng', dpi=300, use_ocrmypdf=True)
    rag = LocalPGVectorRAG(db_config, model_name='mukaj/fin-mpnet-base', ocr_config=ocr_config)
    rag.embedding_model = SentenceTransformer('mukaj/fin-mpnet-base', device='cpu')

    # Generate HyDE text and embedding
    hyde_text = rag.hyde_generator.generate_hypothetical_document(query)
    query_embedding = rag.embedding_model.encode(hyde_text)

    # Build SQL correctly with two placeholders
    sql = """
        SELECT content, pdf_name, pdf_link, year, doc_type, chunk_index, ocr_processed,
               1 - (embedding <=> %s::vector) AS similarity
        FROM financial_documents
        ORDER BY similarity DESC
        LIMIT %s
    """
    params = [query_embedding.tolist(), 5]

    # Execute
    conn = rag.connection_pool.getconn()
    cursor = conn.cursor()
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cursor.close()
    rag.connection_pool.putconn(conn)

    # Display
    print(f"\nTop {len(rows)} chunks for query: \"{query}\"\n" + "="*60)
    for idx, (content, pdf_name, pdf_link, year, doc_type, chunk_index, ocr_processed, similarity) in enumerate(rows, start=1):
        ocr_flag = "[OCR]" if ocr_processed else ""
        print(f"Result {idx} {ocr_flag} | Similarity: {similarity:.3f}")
        print(f"Document: {pdf_name} ({year}, {doc_type})")
        print(f"Link: {pdf_link}")
        print(f"Chunk Index: {chunk_index}\n")
        print(content[:300].strip() + "…\n")
        print("-"*60)

if __name__ == "__main__":
    main()
