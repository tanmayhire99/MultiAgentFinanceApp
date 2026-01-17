#!/usr/bin/env python3
"""
Enhanced RAG System with OCR Support for Scanned PDFs
Includes Tesseract OCR, pdf2image, and OCRmyPDF integration
"""

import os
import sys
import json
import hashlib
import logging
import tempfile
import warnings
import subprocess
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re

import fitz  # pymupdf
import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import requests
import torch

# OCR related imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    logging.warning(f"OCR libraries not available: {e}. Install with: pip install pytesseract pdf2image pillow")

try:
    import ocrmypdf
    OCRMYPDF_AVAILABLE = True
except ImportError:
    OCRMYPDF_AVAILABLE = False
    logging.warning("OCRmyPDF not available. Install with: pip install ocrmypdf")

# Suppress CUDA warnings for unsupported GPUs
warnings.filterwarnings("ignore", message=".*CUDA capability.*")
warnings.filterwarnings("ignore", message=".*Quadro M2200.*")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for financial documents"""
    pdf_name: str
    pdf_link: str
    year: int
    doc_type: str  # Annual report/Transcript/policy/amendments

@dataclass
class ChunkConfig:
    """Configuration for document chunking"""
    chunk_size: int = 500  # words
    overlap_ratio: float = 0.2  # 20% overlap
    min_chunk_size: int = 50  # minimum words per chunk

@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    use_ocr: bool = True
    tesseract_cmd: str = None  # Will auto-detect
    tessdata_dir: str = None   # Will auto-detect
    ocr_language: str = "eng"  # Tesseract language
    dpi: int = 300  # DPI for PDF to image conversion
    use_ocrmypdf: bool = True  # Prefer ocrmypdf over manual approach
    ocr_timeout: int = 300  # OCR timeout in seconds

class HyDEGenerator:
    """Generates hypothetical documents for improved retrieval"""
    
    def __init__(self, api_key: str = None):
        """Initialize HyDE generator with optional API key"""
        self.api_key = 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#api_key or os.getenv('NVIDIA_API_KEY')
        self.api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        
    def generate_hypothetical_document(self, query: str, domain: str = "financial") -> str:
        """Generate a hypothetical document for the given query"""
        
        if not self.api_key:
            logger.warning("No API key provided for HyDE. Using query as-is.")
            return query
            
        prompt = f"""
        Write a detailed financial document that would answer this question: {query}
        
        Focus on providing comprehensive information about:
        - Financial metrics and analysis
        - Market conditions and trends  
        - Investment strategies and recommendations
        - Risk assessments and considerations
        - Regulatory and compliance aspects
        
        Write as if this is an excerpt from a professional financial report or analysis.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a financial analyst writing professional financial documents."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fallback to original query

class OCRProcessor:
    """Handles OCR processing for scanned PDFs"""
    
    def __init__(self, config: OCRConfig = None):
        """Initialize OCR processor"""
        self.config = config or OCRConfig()
        self._setup_tesseract()
        
    def _setup_tesseract(self):
        """Setup Tesseract OCR configuration"""
        if not OCR_AVAILABLE:
            logger.warning("OCR not available. Some PDFs may not be processed.")
            return
            
        # Try to find Tesseract executable
        tesseract_locations = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',  # macOS with Homebrew
            'tesseract'  # System PATH
        ]
        
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        else:
            for location in tesseract_locations:
                try:
                    result = subprocess.run([location, '--version'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        pytesseract.pytesseract.tesseract_cmd = location
                        logger.info(f"Found Tesseract at: {location}")
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            else:
                logger.warning("Tesseract not found. Please install tesseract-ocr")
        
        # Setup tessdata directory
        if self.config.tessdata_dir:
            os.environ['TESSDATA_PREFIX'] = self.config.tessdata_dir
        
    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """Check if PDF is likely scanned (image-based)"""
        try:
            doc = fitz.open(pdf_path)
            
            # Check first few pages
            pages_to_check = min(3, len(doc))
            text_chars = 0
            image_count = 0
            
            for page_num in range(pages_to_check):
                page = doc.load_page(page_num)
                
                # Count text characters
                text = page.get_text()
                text_chars += len(text.strip())
                
                # Count images
                image_list = page.get_images()
                image_count += len(image_list)
            
            doc.close()
            
            # Heuristic: if very little text but images present, likely scanned
            avg_text_per_page = text_chars / pages_to_check
            avg_images_per_page = image_count / pages_to_check
            
            is_scanned = avg_text_per_page < 100 and avg_images_per_page > 0
            
            logger.info(f"PDF analysis - Avg text/page: {avg_text_per_page:.1f}, "
                       f"Avg images/page: {avg_images_per_page:.1f}, "
                       f"Likely scanned: {is_scanned}")
            
            return is_scanned
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {e}")
            return True  # Assume scanned if can't analyze
    
    def _ocr_with_ocrmypdf(self, pdf_path: str) -> str:
        """Use OCRmyPDF to process scanned PDF"""
        if not OCRMYPDF_AVAILABLE:
            raise Exception("OCRmyPDF not available")
            
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            logger.info("Running OCRmyPDF...")
            
            # OCRmyPDF parameters
            ocrmypdf.ocr(
                pdf_path,
                temp_path,
                language=self.config.ocr_language,
                deskew=True,
                clean=True,
                remove_background=False,
                force_ocr=True,
                timeout=self.config.ocr_timeout
            )
            
            # Extract text from OCR'd PDF
            doc = fitz.open(temp_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"
            
            doc.close()
            
            logger.info("OCRmyPDF processing completed successfully")
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCRmyPDF failed: {e}")
            raise
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _ocr_with_tesseract(self, pdf_path: str) -> str:
        """Use Tesseract directly via pdf2image"""
        if not OCR_AVAILABLE:
            raise Exception("OCR libraries not available")
            
        logger.info("Converting PDF to images for OCR...")
        
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.config.dpi,
                fmt='jpeg'
            )
            
            logger.info(f"Converted PDF to {len(images)} images")
            
            # OCR each page
            extracted_text = []
            
            for i, image in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")
                
                # Configure Tesseract
                custom_config = r'--oem 3 --psm 3 -l ' + self.config.ocr_language
                
                # Extract text from image
                text = pytesseract.image_to_string(image, config=custom_config)
                extracted_text.append(text)
            
            # Combine all text
            full_text = "\n".join(extracted_text)
            
            logger.info("Tesseract OCR processing completed successfully")
            return full_text
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR if needed"""
        if not self.config.use_ocr:
            raise Exception("OCR is disabled")
        
        logger.info(f"Starting OCR processing for: {pdf_path}")
        
        # Try OCRmyPDF first (usually better quality)
        if self.config.use_ocrmypdf and OCRMYPDF_AVAILABLE:
            try:
                return self._ocr_with_ocrmypdf(pdf_path)
            except Exception as e:
                logger.warning(f"OCRmyPDF failed, falling back to Tesseract: {e}")
        
        # Fallback to manual Tesseract approach
        if OCR_AVAILABLE:
            return self._ocr_with_tesseract(pdf_path)
        else:
            raise Exception("No OCR method available")

class LocalPGVectorRAG:
    """Local PostgreSQL + pgvector RAG system with OCR support"""
    
    def __init__(self, db_config: Dict[str, Any], model_name: str = "mukaj/fin-mpnet-base", 
                 ocr_config: OCRConfig = None):
        """Initialize the RAG system"""
        self.db_config = db_config
        self.model_name = model_name
        self.chunk_config = ChunkConfig()
        self.hyde_generator = HyDEGenerator()
        self.ocr_processor = OCRProcessor(ocr_config or OCRConfig())
        
        # Initialize embedding model with device selection
        self._setup_device()
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            # Force CPU if CUDA is not compatible
            if not torch.cuda.is_available() or not self._is_cuda_compatible():
                logger.info("Using CPU for embeddings (CUDA not available or incompatible)")
                self.embedding_model = SentenceTransformer(model_name, device='cpu')
            else:
                self.embedding_model = SentenceTransformer(model_name)
                
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.info("Trying to load model on CPU...")
            self.embedding_model = SentenceTransformer(model_name, device='cpu')
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize database connection pool
        self.connection_pool = self._create_connection_pool()
        
        # Setup database schema
        self._setup_database()
        
    def _setup_device(self):
        """Setup device for embeddings"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available. GPU: {gpu_name}")
        else:
            logger.info("CUDA not available. Using CPU.")
    
    def _is_cuda_compatible(self):
        """Check if CUDA is compatible with current GPU"""
        try:
            if not torch.cuda.is_available():
                return False
            
            # Try to run a simple CUDA operation
            torch.cuda.current_device()
            test_tensor = torch.tensor([1.0]).cuda()
            test_result = test_tensor + 1
            return True
        except Exception as e:
            logger.warning(f"CUDA compatibility check failed: {e}")
            return False
    
    def _create_connection_pool(self):
        """Create database connection pool with error handling"""
        try:
            return SimpleConnectionPool(
                1, 20,
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            logger.info("Please check your database configuration and ensure PostgreSQL is running.")
            raise
        
    def _setup_database(self):
        """Setup database tables and indexes with better error handling"""
        conn = None
        cursor = None
        try:
            conn = self.connection_pool.getconn()
            register_vector(conn)
            cursor = conn.cursor()
            
            # Enable pgvector extension (may fail if already exists)
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("pgvector extension enabled")
            except Exception as e:
                logger.warning(f"Could not create vector extension (may already exist): {e}")
            
            # Create documents table
            try:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS financial_documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({self.embedding_dim}),
                        pdf_name VARCHAR(255) NOT NULL,
                        pdf_link TEXT,
                        year INTEGER,
                        doc_type VARCHAR(100),
                        chunk_index INTEGER,
                        content_hash VARCHAR(64),
                        ocr_processed BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(content_hash)
                    );
                """)
                logger.info("Documents table created/verified")
            except psycopg2.errors.InsufficientPrivilege as e:
                logger.error("Insufficient database privileges. Please run the fix_db_permissions.sh script:")
                logger.error("chmod +x fix_db_permissions.sh && ./fix_db_permissions.sh")
                raise
            except Exception as e:
                logger.error(f"Failed to create documents table: {e}")
                raise
            
            # Create indexes
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS financial_documents_embedding_idx 
                    ON financial_documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS financial_documents_metadata_idx 
                    ON financial_documents (pdf_name, year, doc_type);
                """)
                logger.info("Database indexes created/verified")
            except Exception as e:
                logger.warning(f"Could not create indexes (will create later): {e}")
            
            conn.commit()
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database setup failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.connection_pool.putconn(conn)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool]:
        """Extract text from PDF file, with OCR fallback"""
        ocr_used = False
        
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            # First, try regular text extraction
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"
            
            doc.close()
            
            # Check if we got meaningful text
            if text.strip() and len(text.strip()) > 100:
                logger.info("Successfully extracted text using standard method")
                return text.strip(), ocr_used
            
            # If no meaningful text, check if it's a scanned PDF
            logger.info("Little/no text found, checking if PDF needs OCR...")
            
            if self.ocr_processor._is_scanned_pdf(pdf_path):
                logger.info("PDF appears to be scanned, applying OCR...")
                
                try:
                    ocr_text = self.ocr_processor.extract_text_with_ocr(pdf_path)
                    if ocr_text.strip():
                        logger.info("OCR extraction successful")
                        return ocr_text.strip(), True
                    else:
                        logger.warning("OCR completed but no text extracted")
                except Exception as e:
                    logger.error(f"OCR processing failed: {e}")
                    logger.info("Proceeding with standard extraction (may be limited)")
            
            # Return whatever text we have
            return text.strip(), ocr_used
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
            
        words = text.split()
        
        if len(words) <= self.chunk_config.chunk_size:
            return [text]
        
        chunks = []
        overlap_size = int(self.chunk_config.chunk_size * self.chunk_config.overlap_ratio)
        
        for i in range(0, len(words), self.chunk_config.chunk_size - overlap_size):
            chunk_words = words[i:i + self.chunk_config.chunk_size]
            
            if len(chunk_words) >= self.chunk_config.min_chunk_size:
                chunk_text = ' '.join(chunk_words)
                chunks.append(chunk_text)
            
            if i + self.chunk_config.chunk_size >= len(words):
                break
        
        return chunks
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def embed_document(self, pdf_path: str, metadata: DocumentMetadata) -> bool:
        """Process and embed a single document with OCR support"""
        conn = None
        cursor = None
        try:
            logger.info(f"Processing document: {pdf_path}")
            
            # Extract text from PDF (with OCR if needed)
            text, ocr_used = self._extract_text_from_pdf(pdf_path)
            
            if not text:
                logger.error("No text could be extracted from the PDF")
                return False
            
            logger.info(f"Extracted {len(text)} characters (OCR used: {ocr_used})")
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            if not chunks:
                logger.error("No chunks could be created from the text")
                return False
                
            logger.info(f"Created {len(chunks)} chunks")
            
            # Process each chunk
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            embedded_count = 0
            for chunk_index, chunk in enumerate(chunks):
                try:
                    content_hash = self._generate_content_hash(chunk)
                    
                    # Check if chunk already exists
                    cursor.execute(
                        "SELECT id FROM financial_documents WHERE content_hash = %s",
                        (content_hash,)
                    )
                    
                    if cursor.fetchone():
                        logger.debug(f"Chunk {chunk_index} already exists, skipping")
                        continue
                    
                    # Generate embedding
                    logger.debug(f"Generating embedding for chunk {chunk_index + 1}/{len(chunks)}")
                    embedding = self.embedding_model.encode(chunk)
                    
                    # Insert into database
                    cursor.execute("""
                        INSERT INTO financial_documents 
                        (content, embedding, pdf_name, pdf_link, year, doc_type, chunk_index, content_hash, ocr_processed)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        chunk,
                        embedding.tolist(),
                        metadata.pdf_name,
                        metadata.pdf_link,
                        metadata.year,
                        metadata.doc_type,
                        chunk_index,
                        content_hash,
                        ocr_used
                    ))
                    
                    embedded_count += 1
                    
                    # Show progress for large documents
                    if (chunk_index + 1) % 10 == 0:
                        logger.info(f"Processed {chunk_index + 1}/{len(chunks)} chunks")
                        
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_index}: {e}")
                    continue
            
            conn.commit()
            logger.info(f"Successfully embedded {embedded_count} new chunks from {pdf_path}")
            return embedded_count > 0
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Document embedding failed for {pdf_path}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.connection_pool.putconn(conn)
    
    def search_documents(self, query: str, limit: int = 5, use_hyde: bool = True, 
                            year_filter: int = None, doc_type_filter: str = None) -> List[Dict[str, Any]]:
            """Search for relevant documents"""
            conn = None
            cursor = None
            try:
                # Generate HyDE document if enabled
                if use_hyde:
                    search_text = self.hyde_generator.generate_hypothetical_document(query)
                    logger.info("Using HyDE for enhanced retrieval")
                else:
                    search_text = query
                
                # Generate query embedding
                query_embedding = self.embedding_model.encode(search_text)
                
                # Build SQL query with optional filters
                base_query = """
                    SELECT content, pdf_name, pdf_link, year, doc_type, chunk_index, ocr_processed,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM financial_documents
                """
                
                conditions = []
                params = [query_embedding.tolist()]
                
                if year_filter:
                    conditions.append("year = %s")
                    params.append(year_filter)
                
                if doc_type_filter:
                    conditions.append("doc_type = %s")
                    params.append(doc_type_filter)
                
                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)
                
                base_query += " ORDER BY similarity DESC LIMIT %s"
                params.append(limit)
                
                # Execute search
                conn = self.connection_pool.getconn()
                cursor = conn.cursor()
                cursor.execute(base_query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'content': row[0],
                        'pdf_name': row[1],
                        'pdf_link': row[2],
                        'year': row[3],
                        'doc_type': row[4],
                        'chunk_index': row[5],
                        'ocr_processed': row[6],
                        'similarity': float(row[7])
                    })
                
                logger.info(f"Found {len(results)} relevant documents")
                return results
                
            except Exception as e:
                logger.error(f"Document search failed: {e}")
                return []
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self.connection_pool.putconn(conn)

    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        conn = None
        cursor = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT pdf_name) as unique_documents,
                    COUNT(DISTINCT year) as unique_years,
                    COUNT(DISTINCT doc_type) as unique_doc_types,
                    COUNT(CASE WHEN ocr_processed = true THEN 1 END) as ocr_processed_chunks,
                    MIN(year) as earliest_year,
                    MAX(year) as latest_year
                FROM financial_documents
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'total_chunks': row[0],
                    'unique_documents': row[1],
                    'unique_years': row[2],
                    'unique_doc_types': row[3],
                    'ocr_processed_chunks': row[4],
                    'earliest_year': row[5],
                    'latest_year': row[6]
                }
            else:
                return {
                    'total_chunks': 0,
                    'unique_documents': 0,
                    'unique_years': 0,
                    'unique_doc_types': 0,
                    'ocr_processed_chunks': 0,
                    'earliest_year': None,
                    'latest_year': None
                }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.connection_pool.putconn(conn)

def main():
    """Main function for testing the RAG system"""
    # Configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_rag',
        'user': 'tanmay',
        'password': '1999',  # Update this
        'port': 5432
    }
    
    # OCR configuration
    ocr_config = OCRConfig(
        use_ocr=True,
        ocr_language="eng",
        dpi=300,
        use_ocrmypdf=True
    )
    
    try:
        # Initialize RAG system
        rag = LocalPGVectorRAG(db_config, ocr_config=ocr_config)
        
        # Example: Embed a document
        if len(sys.argv) > 1 and sys.argv[1] == "embed":
            if len(sys.argv) < 7:
                print("Usage: python rag_system_ocr.py embed <pdf_path> <pdf_name> <pdf_link> <year> <doc_type>")
                sys.exit(1)
            
            pdf_path = sys.argv[2]
            metadata = DocumentMetadata(
                pdf_name=sys.argv[3],
                pdf_link=sys.argv[4],
                year=int(sys.argv[5]),
                doc_type=sys.argv[6]
            )
            
            success = rag.embed_document(pdf_path, metadata)
            if success:
                print("Document embedded successfully!")
            else:
                print("Document embedding failed!")
        
        # Example: Search documents
        elif len(sys.argv) > 1 and sys.argv[1] == "search":
            if len(sys.argv) < 3:
                print("Usage: python rag_system_ocr.py search <query>")
                sys.exit(1)
            
            query = " ".join(sys.argv[2:])
            results = rag.search_documents(query, limit=3)
            
            print(f"\nSearch results for: {query}")
            print("=" * 50)
            
            if not results:
                print("No results found.")
            else:
                for i, result in enumerate(results, 1):
                    ocr_tag = " [OCR]" if result['ocr_processed'] else ""
                    print(f"\nResult {i} (Similarity: {result['similarity']:.3f}){ocr_tag}")
                    print(f"Document: {result['pdf_name']} ({result['year']})")
                    print(f"Type: {result['doc_type']}")
                    print(f"Content: {result['content'][:200]}...")
        
        # Example: Get statistics
        elif len(sys.argv) > 1 and sys.argv[1] == "stats":
            stats = rag.get_document_stats()
            print("Database Statistics:")
            print("=" * 20)
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        else:
            print("Usage:")
            print("  python rag_system_ocr.py embed <pdf_path> <pdf_name> <pdf_link> <year> <doc_type>")
            print("  python rag_system_ocr.py search <query>")
            print("  python rag_system_ocr.py stats")
            print("\nOCR Support:")
            print(f"  Tesseract OCR: {'Available' if OCR_AVAILABLE else 'Not Available'}")
            print(f"  OCRmyPDF: {'Available' if OCRMYPDF_AVAILABLE else 'Not Available'}")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Install OCR dependencies: sudo apt install tesseract-ocr poppler-utils")
        print("2. Install Python OCR packages: pip install pytesseract pdf2image pillow ocrmypdf")
        print("3. Ensure PostgreSQL is running: sudo systemctl status postgresql")
        print("4. Check database credentials in the code")
        print("5. Run database permission fix: chmod +x fix_db_permissions.sh && ./fix_db_permissions.sh")
        sys.exit(1)

if __name__ == "__main__":
    main()