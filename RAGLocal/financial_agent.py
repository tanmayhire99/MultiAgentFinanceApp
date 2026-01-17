#!/usr/bin/env python3
"""
Financial Agent Conversation System with RAG Integration
Handles conversations, guardrails, and conversation management
"""

import json
import re
import os
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Import the RAG system
from rag_system import LocalPGVectorRAG, DocumentMetadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Individual conversation message"""
    role: str
    content: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None

@dataclass
class ConversationSession:
    """Complete conversation session"""
    session_id: str
    timestamp: str
    model: str
    domain: str
    total_messages: int
    messages: List[ConversationMessage]

class FinancialGuardrails:
    """Enhanced guardrails for financial conversations"""
    
    def __init__(self):
        self.financial_keywords = {
            'investment': ['investment', 'invest', 'portfolio', 'stocks', 'bonds', 'mutual funds', 'etf'],
            'retirement': ['retirement', 'pension', '401k', '403b', 'ira', 'roth', 'social security'],
            'tax': ['tax', 'taxes', 'deduction', 'exemption', 'filing', 'irs', 'taxable'],
            'insurance': ['insurance', 'life insurance', 'health insurance', 'disability', 'coverage'],
            'loans': ['loan', 'mortgage', 'credit', 'debt', 'borrowing', 'lending', 'apr', 'interest'],
            'savings': ['savings', 'emergency fund', 'budgeting', 'financial planning'],
            'banking': ['bank', 'checking', 'savings account', 'cd', 'money market']
        }
    
    def is_financial_query(self, query: str) -> bool:
        """Check if query is financial-related"""
        query_lower = query.lower()
        
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return True
        
        return False
    
    def count_complex_keywords(self, query: str) -> int:
        """Count financial keywords to detect complex queries"""
        query_lower = query.lower()
        keyword_count = 0
        
        for category, keywords in self.financial_keywords.items():
            category_found = False
            for keyword in keywords:
                if keyword in query_lower and not category_found:
                    keyword_count += 1
                    category_found = True
                    break
        
        return keyword_count
    
    def is_too_complex(self, query: str) -> bool:
        """Check if query is too complex (3+ different financial areas)"""
        return self.count_complex_keywords(query) >= 3

class FinancialAgent:
    """Main financial agent with RAG integration"""
    
    def __init__(self, db_config: Dict[str, Any], api_key: str = None):
        """Initialize financial agent"""
        self.api_key = "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"#api_key or os.getenv('NVIDIA_API_KEY')
        self.api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model_name = "meta/llama-3.1-405b-instruct"
        
        # Initialize RAG system
        self.rag_system = LocalPGVectorRAG(db_config)
        
        # Initialize guardrails
        self.guardrails = FinancialGuardrails()
        
        # Conversation storage
        self.conversation_file = "financial_conversations.json"
        self.summary_file = "summarize.json"
        self.conversation_count = 0
        
        # System prompt for financial domain
        self.system_prompt = """You are a helpful financial assistant specializing in personal finance, investments, banking, insurance, loans, retirement planning, and tax matters.

IMPORTANT GUIDELINES:
1. Only answer questions related to financial topics
2. Provide accurate, helpful information while emphasizing the importance of consulting with qualified financial professionals
3. Use the provided context from financial documents to support your answers
4. If you don't have enough information, clearly state this
5. Never provide specific investment advice or guarantee returns
6. Always recommend consulting with certified financial advisors for personalized advice
7. If the Chunks that you got from the RAG system is not useful for the current query, do not cite the sources and do not use those chunks.
Focus on:
- Personal finance education
- General investment principles
- Banking and credit information
- Insurance basics
- Tax planning concepts
- Retirement planning fundamentals

If the query is not financial-related, politely redirect to financial topics."""
    
    def _load_conversations(self) -> List[Dict[str, Any]]:
        """Load existing conversations"""
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
        return []
    
    def _save_conversations(self, conversations: List[Dict[str, Any]]):
        """Save conversations to file"""
        try:
            with open(self.conversation_file, 'w') as f:
                json.dump(conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def _generate_summary(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of conversations"""
        if not conversations:
            return {}
        
        # Extract key metrics
        total_conversations = len(conversations)
        total_messages = sum(len(conv.get('messages', [])) for conv in conversations)
        
        # Count topics discussed
        topics = {
            'investment': 0, 'retirement': 0, 'tax': 0, 'insurance': 0,
            'loans': 0, 'savings': 0, 'banking': 0
        }
        
        recent_queries = []
        
        for conv in conversations[-10:]:  # Last 10 conversations
            for msg in conv.get('messages', []):
                if msg['role'] == 'user':
                    query = msg['content'].lower()
                    recent_queries.append(msg['content'])
                    
                    # Count topics
                    for topic, keywords in self.guardrails.financial_keywords.items():
                        if topic in topics:
                            for keyword in keywords:
                                if keyword in query:
                                    topics[topic] += 1
                                    break
        
        return {
            'generated_at': datetime.now().isoformat(),
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'topics_discussed': topics,
            'most_recent_queries': recent_queries[-5:],  # Last 5 queries
            'conversation_dates': [conv.get('timestamp', '') for conv in conversations[-5:]]
        }
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save conversation summary"""
        try:
            existing_summaries = []
            if os.path.exists(self.summary_file):
                with open(self.summary_file, 'r') as f:
                    existing_summaries = json.load(f)
            
            existing_summaries.append(summary)
            
            with open(self.summary_file, 'w') as f:
                json.dump(existing_summaries, f, indent=2)
                
            logger.info("Conversation summary saved")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def _get_rag_context(self, query: str) -> str:
        """Get relevant context from RAG system"""
        try:
            results = self.rag_system.search_documents(query, limit=3, use_hyde=True)
            
            if not results:
                return "No relevant financial documents found."
            
            context = "Based on financial documents:\n\n"
            for i, result in enumerate(results, 1):
                context += f"{i}. From {result['pdf_name']} ({result['year']}, {result['doc_type']}):\n"
                context += f"{result['content']}\n\n"
            
            return context
        except Exception as e:
            logger.error(f"RAG context retrieval failed: {e}")
            return "Unable to retrieve relevant financial documents."
    
    def _call_llm(self, messages: List[Dict[str, str]], context: str = "") -> str:
        """Call the LLM with context"""
        if not self.api_key:
            return "Error: No API key configured for the financial agent."
        
        # Prepare messages with context
        system_message = self.system_prompt
        if context:
            system_message += f"\n\nRELEVANT FINANCIAL CONTEXT:\n{context}"
        
        llm_messages = [{"role": "system", "content": system_message}] + messages
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": llm_messages,
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"
    
    def process_query(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a financial query with guardrails and RAG"""
        timestamp = datetime.now().isoformat()
        
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Apply guardrails
        if not self.guardrails.is_financial_query(query):
            return {
                'response': "I'm a financial assistant and can only help with questions about personal finance, investments, banking, insurance, loans, retirement planning, and tax matters. Could you please ask a financial question?",
                'sources': [],
                'timestamp': timestamp,
                'guardrail_triggered': 'non_financial'
            }
        
        # Check for complex queries
        if self.guardrails.is_too_complex(query):
            return {
                'response': "Kindly contact Fin Advisor. Your query involves multiple complex financial areas that would benefit from personalized professional consultation.",
                'sources': [],
                'timestamp': timestamp,
                'guardrail_triggered': 'too_complex'
            }
        
        # Get RAG context
        context = self._get_rag_context(query)
        
        # Prepare conversation for LLM
        messages = conversation_history + [{"role": "user", "content": query}]
        
        # Get LLM response
        response = self._call_llm(messages, context)
        
        # Extract source information
        sources = []
        try:
            rag_results = self.rag_system.search_documents(query, limit=3, use_hyde=True)
            for result in rag_results:
                sources.append({
                    'pdf_name': result['pdf_name'],
                    'pdf_link': result['pdf_link'],
                    'year': result['year'],
                    'doc_type': result['doc_type'],
                    'similarity': result['similarity']
                })
        except Exception as e:
            logger.error(f"Failed to extract sources: {e}")
        
        return {
            'response': response,
            'sources': sources,
            'timestamp': timestamp,
            'context_used': bool(context and "No relevant" not in context),
            'guardrail_triggered': None
        }
    
    def start_conversation(self) -> str:
        """Start a new conversation session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting new conversation session: {session_id}")
        return session_id
    
    def save_conversation_turn(self, session_id: str, user_message: str, 
                             agent_response: Dict[str, Any]):
        """Save a conversation turn"""
        try:
            conversations = self._load_conversations()
            
            # Find or create conversation session
            session = None
            for conv in conversations:
                if conv.get('session_id') == session_id:
                    session = conv
                    break
            
            if not session:
                session = {
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model_name,
                    'domain': 'financial',
                    'messages': []
                }
                conversations.append(session)
            
            # Add user message
            session['messages'].append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Add agent response
            session['messages'].append({
                'role': 'assistant',
                'content': agent_response['response'],
                'timestamp': agent_response['timestamp'],
                'sources': agent_response.get('sources', [])
            })
            
            # Update session info
            session['total_messages'] = len(session['messages'])
            session['last_updated'] = datetime.now().isoformat()
            
            # Save conversations
            self._save_conversations(conversations)
            
            # Check if we need to create a summary (every 5 conversations)
            self.conversation_count += 1
            if self.conversation_count % 5 == 0:
                summary = self._generate_summary(conversations)
                self._save_summary(summary)
            
            logger.info(f"Conversation turn saved for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation turn: {e}")

def main():
    """Main function for running the financial agent"""
    # Configuration
    db_config = {
        'host': 'localhost',
        'database': 'financial_rag',
        'user': 'tanmay',
        'password': '1999',
        'port': 5432
    }
    
    # Initialize agent
    agent = FinancialAgent(db_config)
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        session_id = agent.start_conversation()
        conversation_history = []
        
        print("Financial Agent Chat (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                if not user_input:
                    continue
                
                # Process query
                response = agent.process_query(user_input, conversation_history)
                
                # Display response
                print(f"\nAgent: {response['response']}")
                
                if response.get('sources'):
                    print("\nSources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['pdf_name']} ({source['year']}) - {source['doc_type']}")
                
                # Save conversation
                agent.save_conversation_turn(session_id, user_input, response)
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response['response']})
                
                # Keep last 10 messages for context
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    # Single query mode
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        if len(sys.argv) < 3:
            print("Usage: python financial_agent.py query <your_question>")
            return
        
        query = " ".join(sys.argv[2:])
        response = agent.process_query(query)
        
        print(f"Query: {query}")
        print(f"Response: {response['response']}")
        
        if response.get('sources'):
            print("\nSources:")
            for source in response['sources']:
                print(f"- {source['pdf_name']} ({source['year']})")
    
    else:
        print("Usage:")
        print("  python financial_agent.py interactive  # Start interactive chat")
        print("  python financial_agent.py query <question>  # Single question")

if __name__ == "__main__":
    import sys
    main()