#!/usr/bin/env python3
"""
Minimal Enhanced RAG Pipeline with HyDE
For indexing PDFs into PostgreSQL + pgvector
"""

import os
import hashlib
import json
import tempfile

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import fitz  # pymupdf
import ocrmypdf
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Optional: OpenAI for HyDE
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class HyDEConfig:
    enabled: bool = True
    backend: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 300
    temperature: float = 0.7
    fallback_to_original: bool = True
    cache_responses: bool = True


class HyDEQueryTranslator:
    def __init__(self, config: HyDEConfig):
        self.config = config
        self.cache = {}
        self.logger = logging.getLogger(__name__ + ".hyde")

    def generate_hypothetical_document(self, query: str) -> str:
        if not self.config.enabled or self.config.backend != "openai" or not HAS_OPENAI:
            return query
        cache_key = hashlib.md5(f"{query}_{self.config.model_name}".encode()).hexdigest()
        if self.config.cache_responses and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a Finance expert."},
                    {"role": "user", "content": f"Give a professional answer to the financial query: {query}"}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            text = response.choices[0].message.content.strip()
            if self.config.cache_responses:
                self.cache[cache_key] = text
            return text
        except Exception as e:
            self.logger.error(f"HyDE failed: {e}")
            if self.config.fallback_to_original:
                return query
            else:
                raise


class EnhancedLocalPDFRAGPipeline:
    def __init__(self, db_config: dict, pdf_path: str, hyde_config: Optional[HyDEConfig] = None):
        self.db_config = db_config
        self.pdf_path = pdf_path
        self.embedding_model = SentenceTransformer('mukaj/fin-mpnet-base') #baconnier/Finance_embedding_large_en-V1.5
        self.connection_pool: Optional[SimpleConnectionPool] = None
        self.hyde_config = hyde_config or HyDEConfig()
        self.hyde_translator = HyDEQueryTranslator(self.hyde_config)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def create_connection_pool(self):
        try:
            self.connection_pool = SimpleConnectionPool(minconn=1, maxconn=5, **self.db_config)
            self.logger.info("Database connection pool created")
        except psycopg2.Error as e:
            self.logger.error(f"Connection pool failed: {e}")

    def get_connection(self):
        if not self.connection_pool:
            self.create_connection_pool()
        conn = self.connection_pool.getconn()
        register_vector(conn)
        return conn

    def return_connection(self, conn):
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)

    def close_connections(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("All connections closed")

    def get_file_hash(self):
        hash_md5 = hashlib.md5()
        with open(self.pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def detect_pdf_type(self):
        try:
            doc = fitz.open(self.pdf_path)
            total_chars = sum(len(doc[page].get_text().strip()) for page in range(min(3, len(doc))))
            doc.close()
            avg_chars = total_chars / 3 if total_chars else 0
            return "scanned" if avg_chars < 100 else "native"
        except:
            return "native"

    def apply_ocr_if_needed(self):
        if self.detect_pdf_type() == "scanned":
            temp_dir = tempfile.mkdtemp()
            ocr_output = os.path.join(temp_dir, "ocr_output.pdf")
            ocrmypdf.ocr(self.pdf_path, ocr_output, deskew=True, skip_text=True)
            self.pdf_path = ocr_output

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(texts)

    def chunk_text(self, text, max_chunk=1000, overlap=200):
        chunks = []
        current = ""
        for paragraph in text.split("\n\n"):
            if len(current + paragraph) > max_chunk and current:
                chunks.append(current.strip())
                current = current[-overlap:] + paragraph
            else:
                current += paragraph
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def generate_embeddings_batch(self, chunks):
        return self.embedding_model.encode(chunks, convert_to_numpy=True).tolist()

    def store_document_and_chunks(self, pdf_path, chunks, embeddings):
        if len(chunks) != len(embeddings):
            self.logger.error("Chunks and embeddings length mismatch")
            return None

        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            filename = os.path.basename(pdf_path)
            file_size = os.path.getsize(pdf_path)
            file_hash = self.get_file_hash()

            # Insert document
            cursor.execute("""
                INSERT INTO documents (filename, file_path, file_size, file_hash, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                RETURNING id
            """, (filename, pdf_path, file_size, file_hash))
            doc_id = cursor.fetchone()[0]

            # Insert chunks
            chunk_data = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_timestamp": datetime.now().isoformat()
                }
                chunk_data.append((doc_id, i, chunk, chunk, json.dumps(metadata), emb, len(chunk.split()), len(chunk)))
            cursor.executemany("""
                INSERT INTO document_chunks
                (document_id, chunk_index, content, cleaned_content, chunk_metadata, embedding, word_count, char_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, chunk_data)
            conn.commit()
            self.logger.info(f"Stored {len(chunks)} chunks for document ID {doc_id}")
            return doc_id
        except Exception as e:
            self.logger.error(f"Store failed: {e}")
            conn.rollback()
            return None
        finally:
            cursor.close()
            self.return_connection(conn)
