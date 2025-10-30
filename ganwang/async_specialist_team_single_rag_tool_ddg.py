import os
from dotenv import load_dotenv
load_dotenv()
import time
import json
import re
import datetime
import logging
import sys
import argparse
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout, redirect_stderr
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.chunking.fixed import FixedSizeChunking
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()
llm_dbt_env_path = os.path.join(os.getcwd(), "LLM-DBT", ".env")
if os.path.exists(llm_dbt_env_path):
    load_dotenv(llm_dbt_env_path)

# OpenAI via LiteLLM is used
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set OPENAI_API_KEY in your environment or in the .env file.")

def strip_ansi_codes(text):
    """remove ANSI escape sequences from text to make logs more readable"""
    if not isinstance(text, str):
        return text
        
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def strip_thinking_tokens(text):
    """remove thinking tokens from LLM outputs"""
    if not isinstance(text, str):
        return text
    

    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
    text = think_pattern.sub('', text).strip() 
    
    return text

class TeeLogger:
    """duplicates output to both a file and the original stream"""
    def __init__(self, filename, mode='w', encoding='utf-8', stream=None):
        self.file = open(filename, mode, encoding=encoding)
        self.stream = stream if stream else sys.stdout
        self.encoding = encoding
        
    def write(self, data):
        clean_data = strip_ansi_codes(data)
        self.file.write(clean_data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
        
    def flush(self):
        self.file.flush()
        self.stream.flush()
        
    def close(self):
        if self.file:
            self.file.close()

class InvestmentOffer(BaseModel):
    Valuation: str = Field(..., description="Estimated company valuation (e.g., $10 million)")
    Equity_Offered: str = Field(..., description="Percentage of equity offered to investors (e.g., 10%)")
    Funding_Amount: str = Field(..., description="The amount of funding requested (e.g., $1 million)")
    Key_Terms: str = Field("None", description="Any additional key terms (optional)")

class SharkTankPitch(BaseModel):
    Pitch: str = Field(..., description="The complete investment pitch text")
    Initial_Offer: InvestmentOffer = Field(..., description="The investment offer details")

def add_facts_json_to_knowledge_base(knowledge_base, facts_json_path: str = None, force_reload: bool = False):
    """Add facts from all_processed_facts.json to the knowledge base for RAG"""
    if facts_json_path is None:
        # Try multiple possible locations
        possible_paths = [
            "./data/all_processed_facts.json",
            "./ganwang/all_processed_facts.json",
            os.path.join(os.path.dirname(__file__), "all_processed_facts.json")
        ]
        facts_json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                facts_json_path = path
                break
    
    if not facts_json_path or not os.path.exists(facts_json_path):
        logging.warning(f"Facts JSON file not found. Skipping facts embedding. Tried: {possible_paths}")
        return
    
    try:
        logging.info(f"Loading facts from: {facts_json_path}")
        with open(facts_json_path, 'r', encoding='utf-8') as f:
            all_facts = json.load(f)
        
        if not all_facts:
            logging.warning("Facts JSON file is empty. Skipping facts embedding.")
            return
        
        logging.info(f"Converting {len(all_facts)} product entries into embeddable text...")
        
        # Convert each product entry into a structured text document
        for product_key, product_data in all_facts.items():
            if not isinstance(product_data, dict):
                continue
            
            # Build a comprehensive text representation of the product
            text_parts = []
            
            # Product identification
            text_parts.append(f"PRODUCT: {product_key}")
            
            # Facts
            if 'facts' in product_data:
                facts = product_data['facts']
                if isinstance(facts, dict):
                    text_parts.append("\nFACTS:")
                    for key, value in facts.items():
                        text_parts.append(f"  {key}: {value}")
                else:
                    text_parts.append(f"\nFACTS: {facts}")
            
            # Product Description
            if 'product_description' in product_data:
                pd = product_data['product_description']
                if isinstance(pd, dict):
                    text_parts.append("\nPRODUCT DESCRIPTION:")
                    for key, value in pd.items():
                        text_parts.append(f"  {key}: {value}")
                else:
                    text_parts.append(f"\nPRODUCT DESCRIPTION: {pd}")
            
            # Pitch Summary
            if 'pitch_summary' in product_data:
                ps = product_data['pitch_summary']
                if isinstance(ps, dict):
                    text_parts.append("\nPITCH SUMMARY:")
                    for key, value in ps.items():
                        text_parts.append(f"  {key}: {value}")
                else:
                    text_parts.append(f"\nPITCH SUMMARY: {ps}")
            
            # Initial Offer
            if 'initial_offer' in product_data:
                io = product_data['initial_offer']
                if isinstance(io, dict):
                    text_parts.append("\nINITIAL OFFER:")
                    for key, value in io.items():
                        text_parts.append(f"  {key}: {value}")
                else:
                    text_parts.append(f"\nINITIAL OFFER: {io}")
            
            # Final Offer
            if 'final_offer' in product_data:
                fo = product_data['final_offer']
                if isinstance(fo, dict):
                    text_parts.append("\nFINAL OFFER:")
                    for key, value in fo.items():
                        text_parts.append(f"  {key}: {value}")
                else:
                    text_parts.append(f"\nFINAL OFFER: {fo}")
            
            # Combine into full text
            full_text = "\n".join(text_parts)
            
            # Add to knowledge base
            try:
                knowledge_base.add_content(
                    text_content=full_text,
                    name=product_key,
                    description=f"Product facts and pitch data for {product_key}",
                    metadata={"source": "all_processed_facts.json", "product_key": product_key},
                    skip_if_exists=not force_reload
                )
            except Exception as e:
                logging.warning(f"Error adding product {product_key} to knowledge base: {e}")
                continue
        
        logging.info(f"Successfully added {len(all_facts)} product entries to knowledge base.")
        
    except Exception as e:
        logging.error(f"Error loading facts JSON into knowledge base: {e}")
        logging.exception(e)

def load_knowledge_base(
    pdf_urls: list[str] = None,
    pdf_paths: list[str] = None,
    chunking_strategy_name: str = "fixed",
    db_path: str = "./chromadb_data",
    storage_file: str = "tmp/pdf_rag.db",
    force_reload: bool = False,
    skip_embedding_check: bool = False,
    include_facts_json: bool = True
):
    """set up and return the RAG knowledge base"""
    # Use environment variables if defined (for sharing config across instances)
    env_chunking_strategy = os.environ.get("ASYNC_RAG_CHUNKING_STRATEGY")
    env_db_path = os.environ.get("ASYNC_RAG_DB_PATH")
    env_collection_name = os.environ.get("ASYNC_RAG_COLLECTION_NAME")
    env_force_reload = os.environ.get("ASYNC_RAG_FORCE_RELOAD", "").lower() == "true"
    env_skip_check = os.environ.get("ASYNC_RAG_SKIP_EMBEDDING_CHECK", "").lower() == "true"
    
    if env_chunking_strategy:
        chunking_strategy_name = env_chunking_strategy
    if env_db_path:
        db_path = env_db_path
    if env_force_reload:
        force_reload = True
    if env_skip_check:
        skip_embedding_check = True
    

    if pdf_paths is None and pdf_urls is None:
        # Default to local paths if available, otherwise use URLs
        if os.path.exists("data/pdfs/hbs_opportunities.pdf") and os.path.exists("data/pdfs/hbs_pitchdeck_sample.pdf"):
            pdf_paths = [
                "data/pdfs/hbs_opportunities.pdf",
                "data/pdfs/hbs_pitchdeck_sample.pdf"
            ]
        else:
            pdf_urls = [
                "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_opportunities.pdf",
                "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_pitchdeck_sample.pdf",
            ]

    # Chunking strategies
    chunking_options = {
        "fixed": FixedSizeChunking(),
        "agentic": AgenticChunking(),
        "semantic": SemanticChunking(),
    }

    chunking_strategy = chunking_options.get(chunking_strategy_name, FixedSizeChunking())

    # Database naming
    collection_name = env_collection_name if env_collection_name else f"HBS_{chunking_strategy_name}"

    # Log setup information
    logging.info(f"Knowledge Base Setup:")
    logging.info(f"- Chunking Strategy: {chunking_strategy_name}")
    logging.info(f"- ChromaDB Collection: {collection_name}")
    logging.info(f"- ChromaDB Path: {db_path}")
    
    if pdf_paths:
        logging.info(f"- Local PDFs: {', '.join(pdf_paths)}")
    elif pdf_urls:
        logging.info(f"- PDF URLs: {', '.join(pdf_urls)}")


    os.makedirs(db_path, exist_ok=True)
    
    #  simple approach 
    logging.info("Using default ChromaDB embedding function")

    # persistent storage 
    vector_db = ChromaDb(
        collection=collection_name,
        path=db_path,
        persistent_client=True
    )

    # Check if we already have docs in collection FIRST
    existing_docs = False
    
    # skip check if requested
    if skip_embedding_check:
        logging.info("Skipping embedding check as requested.")
        existing_docs = False
    else:
        try:
            # try different ways to count docs in chromadb
            if hasattr(vector_db, 'collection') and vector_db.collection:
                try:
                    count = vector_db.collection.count()
                    existing_docs = count > 0
                    logging.info(f"Found {count} existing documents in the collection")
                except Exception as e1:
                    logging.debug(f"Method 1 failed: {e1}")
                    
                    # try querying to check
                    try:
                        results = vector_db.search("", n_results=1)
                        has_results = results and len(results) > 0
                        existing_docs = has_results
                        logging.info(f"Collection has documents: {existing_docs}")
                    except Exception as e2:
                        logging.debug(f"Method 2 failed: {e2}")
                        
                        # last resort - peek at collection
                        try:
                            results = vector_db._client.get_collection(vector_db.collection_name)
                            if results:
                                existing_docs = True
                                logging.info("Collection exists and likely has documents")
                        except Exception as e3:
                            logging.debug(f"Method 3 failed: {e3}")
                            existing_docs = False
            else:
                logging.info("No collection available to check")
                existing_docs = False
        except Exception as e:
            logging.info(f"Unable to check for existing documents: {e}. Will load from scratch.")
            existing_docs = False
    
    # create knowledge base using new agno 2.0+ API
    knowledge_base = Knowledge(
        name=collection_name,
        vector_db=vector_db
    )
    
    # Only add content if we don't have existing docs or force reload is requested
    if not existing_docs or force_reload:
        if force_reload:
            logging.info("Force reload requested. Reloading all documents...")
        else:
            logging.info("No existing documents found. Loading and embedding from scratch...")
        
        # Add content based on source type
        if pdf_paths:
            logging.info(f"Adding {len(pdf_paths)} local PDF file(s)")
            for pdf_path in pdf_paths:
                logging.info(f"Adding PDF: {pdf_path}")
                try:
                    knowledge_base.add_content(
                        path=pdf_path,
                        skip_if_exists=not force_reload
                    )
                except Exception as e:
                    logging.error(f"Error adding PDF {pdf_path}: {e}")
                    raise
        elif pdf_urls:
            logging.info(f"Adding {len(pdf_urls)} PDF URL(s)")
            for pdf_url in pdf_urls:
                logging.info(f"Adding PDF URL: {pdf_url}")
                try:
                    knowledge_base.add_content(
                        url=pdf_url,
                        skip_if_exists=not force_reload
                    )
                except Exception as e:
                    logging.error(f"Error adding PDF URL {pdf_url}: {e}")
                    raise
        
        logging.info("Documents loaded and embedded successfully.")
        
        # Add facts JSON to knowledge base if requested
        if include_facts_json:
            logging.info("Adding facts from all_processed_facts.json to knowledge base...")
            add_facts_json_to_knowledge_base(knowledge_base, force_reload=force_reload)
    else:
        logging.info(f"Using existing embedded documents. No need to reload.")
        # Still add facts if they're not there (they might not have been in previous runs)
        if include_facts_json:
            logging.info("Checking if facts JSON needs to be added to knowledge base...")
            add_facts_json_to_knowledge_base(knowledge_base, force_reload=False)
    
    return knowledge_base 

class AsyncSpecialistTeamRAGDDG:
    """coordinates a multi-agent team for SharkTank pitch generation using an asynchronous workflow with both RAG and DuckDuckGo search tools"""
    
    def __init__(self, output_dir, log_file_path, model_id="gpt-4o", max_tokens=4096, temperature=0.7, debug_mode=True):
        """set up the team with output directory, logging, and model parameters"""
        self.logger = logging.getLogger(f"async_team_rag_ddg_{os.path.basename(output_dir)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # OpenAI via LiteLLM is used
        if not os.getenv("OPENAI_API_KEY"):
            self.logger.warning("OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your environment variables.")
        
        self.model_id = model_id
        self.debug_mode = debug_mode
        self.logger.info(f"Using model: {self.model_id}")
        self.logger.info(f"Debug mode: {'enabled' if self.debug_mode else 'disabled'}")
        
        self.log_file_path = log_file_path
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== ASYNC SPECIALIST TEAM WITH RAG AND DUCKDUCKGO (Model: {self.model_id}) ===\n\n")
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger.info(f"Using max_tokens: {self.max_tokens}, temperature: {self.temperature}")
        
        # initialize knowledge base using same approach as single rag
        self.logger.info("Initializing RAG knowledge base...")
        
        # check if we can reuse existing knowledge base from env vars to avoid re-embedding
        self.kb = load_knowledge_base()
        
        # set up ddg search with limited results to handle rate limits
        self.ddg_tools = DuckDuckGoTools(fixed_max_results=3)
        self.logger.info("DuckDuckGo search tools initialized")
        
        def create_openai_model():
            # Use OpenAI models directly via LiteLLM
            # Map common Groq model names to OpenAI equivalents
            model_map = {
                "llama-3.3-70b-versatile": "gpt-4o",
                "deepseek-r1-distill-llama-70b": "gpt-4o",
                "llama3-70b-8192": "gpt-4o",
                "mistral-saba-24b": "gpt-4o",
                "gpt-4": "gpt-4",
                "gpt-3.5-turbo": "gpt-3.5-turbo",
                "gpt-4o": "gpt-4o",
                "o1-mini": "o1-mini",
                "o1-preview": "o1-preview"
            }
            model_to_use = model_map.get(self.model_id, "gpt-4o")
            return LiteLLM(
                id=model_to_use,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        
        # Tool instructions for specialist agents
        tool_usage_instructions = """
        IMPORTANT TOOL USAGE INSTRUCTIONS:
        You must use the DuckDuckGo Search tool EXACTLY ONCE during your analysis - no more, no less.
        
        When you use the search tool, follow this pattern:
        
        I'll use the DuckDuckGo Search tool to find information about [topic].
        
        DO NOT use any special formatting or syntax when referring to the tool.
        If you get a rate limit error, try a different search query.
        
        The search results will provide you with up-to-date, factual information that will enhance your analysis.
        Choose your search query carefully to get the most relevant information for your analysis.
        """
        
        # set up the team of specialist agents with DuckDuckGo search tool
        self.financial_strategist = Agent(
            name="Financial Strategist",
            model=create_openai_model(),
            description="You are a Financial Strategist with expertise in startup valuation, investment analysis, and business model evaluation.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating the financial aspects of a product for a SharkTank pitch.",
                "Always respond in English only.",
                "Use DuckDuckGo Search EXACTLY ONCE to research additional financial information such as:",
                "- Industry standard valuations for similar products",
                "- Recent funding trends in the product's sector", 
                "- Comparable companies and their financial metrics",
                "- Typical investment terms for similar businesses",
                "Choose your ONE search query carefully to get the most valuable financial information.",
                "Analyze the provided facts and product description to develop a financial strategy for a SharkTank pitch, including:",
                "1. A justified valuation for the company",
                "2. An appropriate investment amount to request",
                "3. A fair equity percentage to offer",
                "4. A breakdown of how the funds will be used",
                "5. A realistic ROI timeline",
                "6. Potential exit strategies",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Do not invent or assume financial data that contradicts what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your recommendations."
            ],
            tools=[self.ddg_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        self.market_research_specialist = Agent(
            name="Market Research Specialist",
            model=create_openai_model(),
            description="You are a Market Research Specialist with deep knowledge of consumer trends, market analysis, and competitive landscapes.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating the market aspects of a product for a SharkTank pitch.",
                "Always respond in English only.",
                "Use DuckDuckGo Search EXACTLY ONCE to research additional market information such as:",
                "- Current market size and growth projections",
                "- Emerging trends in the product's industry",
                "- Competitors and their market positions",
                "- Consumer behavior and preferences",
                "Choose your ONE search query carefully to get the most valuable market information.",
                "Analyze the provided facts and product description to develop market insights for a SharkTank pitch, including:",
                "1. The estimated size of the target market",
                "2. Description of target customer segments",
                "3. Analysis of competitors and their strengths/weaknesses",
                "4. Relevant market trends",
                "5. Potential growth opportunities",
                "6. Challenges in the market",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Do not make up market sizes or competitor information that contradicts what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your insights."
            ],
            tools=[self.ddg_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        self.product_technical_advisor = Agent(
            name="Product Technical Advisor",
            model=create_openai_model(),
            description="You are a Product/Technical Advisor with expertise in product development, technical feasibility, and innovation assessment.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating the product/technical aspects for a SharkTank pitch.",
                "Always respond in English only.",
                "Use DuckDuckGo Search EXACTLY ONCE to research additional technical information such as:",
                "- Technical specifications for similar products",
                "- Manufacturing or production processes",
                "- Materials or components",
                "- Intellectual property considerations",
                "- Emerging technologies in the field",
                "Choose your ONE search query carefully to get the most valuable technical information.",
                "Analyze the provided facts and product description to develop product insights for a SharkTank pitch, including:",
                "1. Key product features to highlight",
                "2. Technical advantages over competitors",
                "3. How to effectively demonstrate the product",
                "4. Assessment of production/technical scalability",
                "5. Potential future product developments",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Do not invent capabilities or exaggerate performance in ways that contradict what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your assessments."
            ],
            tools=[self.ddg_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        self.shark_psychology_expert = Agent(
            name="Shark Psychology Expert",
            model=create_openai_model(),
            description="You are a Shark Psychology Expert who understands the motivations, preferences, and decision patterns of SharkTank investors.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating investor psychology for a SharkTank pitch.",
                "Always respond in English only.",
                "Use DuckDuckGo Search EXACTLY ONCE to research additional information such as:",
                "- Recent investments made by Sharks",
                "- Investment preferences of individual Sharks",
                "- Successful negotiation tactics used on the show",
                "- Common reasons for Shark rejections",
                "Choose your ONE search query carefully to get the most valuable investor psychology information.",
                "Analyze the provided facts and product description to develop investor psychology insights for a SharkTank pitch, including:",
                "1. Points that will appeal to Sharks",
                "2. Potential objections and how to counter them",
                "3. Strategy for negotiating with Sharks",
                "4. Tips for effective presentation",
                "5. Sharks that might be the best fit and why",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Focus on general Shark psychology and preferences rather than making specific predictions that contradict what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your insights."
            ],
            tools=[self.ddg_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        # Pitch Drafter - without RAG or search tools
        self.pitch_drafter = Agent(
            name="Pitch Drafter",
            model=create_openai_model(),
            description="You are a skilled pitch writer for entrepreneurs appearing on Shark Tank.",
            instructions=[
                "You are responsible for drafting a SharkTank pitch based on specialist analyses.",
                "Always respond in English only.",
                "Your task is to create a compelling pitch based on the specialist analyses provided. You will receive:",
                "1. The original product facts and description",
                "2. Financial analysis from the Financial Strategist",
                "3. Market research analysis from the Market Research Specialist",
                "4. Product/technical analysis from the Product Technical Advisor",
                "5. Shark psychology insights from the Shark Psychology Expert",
                "The pitch should be structured to grab attention, clearly explain the product/service, highlight market potential, showcase competitive advantages, present financial data, make a specific investment ask, and close with a strong call to action."
            ],
            response_model=SharkTankPitch,
            debug_mode=self.debug_mode
        )
        
        # Pitch Critic with RAG capabilities - using EXACT configuration from async_specialist_team_single_rag.py
        # But with explicit tool setup
        self.pitch_critic = Agent(
            name="Pitch Critic",
            model=create_openai_model(),
            description="You are a Pitch Critic who identifies strengths, weaknesses, and areas for improvement in SharkTank pitches with access to a knowledge base of pitch examples and best practices.",
            instructions=[
                "You are responsible for critiquing a SharkTank pitch draft.",
                "Always respond in English only.",
                "You will be provided with:",
                "1. The original product facts and description",
                "2. The draft pitch in JSON format",
                "You have access to a knowledge base containing best practices for creating pitch decks and identifying business opportunities.",
                "Search this knowledge base for relevant information when critiquing the pitch.",
                "You can use the search_knowledge_base tool to find information in the knowledge base.",
                "Analyze the draft pitch provided and offer constructive criticism to make it more compelling and effective.",
                "Be specific in your feedback and suggest concrete improvements.",
                "Focus on:",
                "1. Strengths of the pitch that should be maintained or emphasized",
                "2. Weaknesses or aspects that could undermine the pitch's effectiveness",
                "3. Specific areas that need improvement (clarity, structure, persuasiveness, etc.)",
                "4. Concrete suggestions for improving the pitch",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your critiques."
            ],
            tools=[], # No additional tools, only knowledge search
            debug_mode=self.debug_mode,
            knowledge=self.kb,
            search_knowledge=True,
            add_references=True
        )
        
        # Pitch Finalizer - without RAG or search tools
        self.pitch_finalizer = Agent(
            name="Pitch Finalizer",
            model=create_openai_model(),
            description="You are a pitch finalization expert for entrepreneurs appearing on Shark Tank.",
            instructions=[
                "You are responsible for finalizing a SharkTank pitch based on critic feedback.",
                "Always respond in English only.",
                "Your task is to refine and finalize the draft pitch based on the critic's feedback.",
                "You will be provided with:",
                "1. The original product facts and description",
                "2. The draft pitch in JSON format",
                "3. The pitch critic's feedback",
                "Create a polished, compelling final pitch that incorporates the strengths identified by the critic while addressing the areas for improvement. The final pitch should be concise, engaging, and strategically structured to maximize appeal to the Sharks."
            ],
            response_model=SharkTankPitch,
            debug_mode=self.debug_mode
        )
        
        self.logger.info("Async Specialist Team with both RAG and DuckDuckGo search tools initialized")
    
    def log_message(self, agent_name, message):
        """log a message from an agent to the consolidated log file"""
        clean_message = strip_ansi_codes(message)
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {agent_name} ===\n\n")
            f.write(clean_message)
    
    async def run_agent_async(self, agent, agent_name, context):
        """run an agent with given context and handle its response asynchronously"""
        self.logger.info(f"{agent_name} analyzing...")
        
        try:
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                # run agent in separate thread since it can block
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: agent.run(context))
                
            debug_output = f.getvalue()
            
            debug_output = strip_ansi_codes(debug_output)
            
            if debug_output:
                self.log_message(f"{agent_name} Debug Output", debug_output)
            
            if response:
                if hasattr(response, 'content'):
                    if hasattr(response.content, 'model_dump'):
                        content_for_conversation = json.dumps(response.content.model_dump(), indent=2)
                    else:
                        content_for_conversation = strip_thinking_tokens(strip_ansi_codes(str(response.content)))
                else:
                    content_for_conversation = strip_thinking_tokens(strip_ansi_codes(str(response)))
                
                if hasattr(response, 'content'):
                    if hasattr(response.content, 'model_dump'):
                        response_str = json.dumps(response.content.model_dump(), indent=2)
                    else:
                        response_str = strip_thinking_tokens(strip_ansi_codes(str(response.content)))
                else:
                    response_str = strip_thinking_tokens(strip_ansi_codes(str(response)))
                    
                response_length = len(response_str)
                self.logger.info(f"{agent_name} response length: {response_length}")
                
                # track both successful and rate-limited tool attempts
                tool_calls = []
                tool_attempts = 0
                
                # check standard locations for tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_calls = response.tool_calls
                elif hasattr(response, 'metrics') and hasattr(response.metrics, 'tool_calls') and response.metrics.tool_calls:
                    tool_calls = response.metrics.tool_calls
                
                # check debug output for failed tool calls
                if debug_output:
                    # count both successful and failed calls
                    tool_attempts_in_debug = debug_output.count("Tool call Id:")
                    rate_limit_errors = debug_output.count("202 Ratelimit")
                    
                    if tool_attempts_in_debug > 0:
                        tool_attempts = tool_attempts_in_debug
                        self.logger.info(f"{agent_name} attempted {tool_attempts} tool calls (from debug output)")
                        
                        if rate_limit_errors > 0:
                            self.logger.warning(f"{agent_name} encountered {rate_limit_errors} rate limit errors during searches")
                
                if len(tool_calls) > 0:
                    self.logger.info(f"{agent_name} made {len(tool_calls)} successful tool calls")
                    
                    # warn if specialist made more than one search
                    if agent_name in ["Financial Strategist", "Market Research Specialist", "Product Technical Advisor", "Shark Psychology Expert"] and len(tool_calls) > 1:
                        self.logger.warning(f"{agent_name} made {len(tool_calls)} tool calls - more than the required exactly one call")
                    
                    for i, tool_call in enumerate(tool_calls):
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        tool_input = getattr(tool_call, 'input', {})
                        tool_output = getattr(tool_call, 'output', {})
                        
                        # log tool calls to consolidated log
                        self.log_message(
                            f"{agent_name} Tool Call #{i+1}",
                            f"Tool: {tool_name}\nQuery: {tool_input}\nResults: {tool_output}"
                        )
                
                # check if pitch critic used RAG
                rag_used = False
                if agent_name == "Pitch Critic" and hasattr(response, 'knowledge_results'):
                    rag_used = True
                    self.logger.info(f"{agent_name} used RAG with {len(response.knowledge_results) if response.knowledge_results else 0} results")
                    
                    # log RAG results
                    if response.knowledge_results:
                        self.log_message(
                            f"{agent_name} RAG Results",
                            f"Knowledge results: {response.knowledge_results}"
                        )
                
                # use max count between tracked calls, debug output and content mentions
                actual_tool_calls = max(len(tool_calls) if tool_calls else 0, tool_attempts)
                
                return {
                    "response": response, 
                    "content": content_for_conversation,
                    "tool_calls": tool_calls if tool_calls else [],
                    "tool_attempts": actual_tool_calls,
                    "rag_used": rag_used
                }
            else:
                self.logger.warning(f"{agent_name} returned empty response")
                error_message = f"No response from {agent_name}"
                self.log_message(agent_name, error_message)
                return {"response": error_message, "content": error_message, "tool_calls": [], "tool_attempts": 0, "rag_used": False}
        except Exception as e:
            self.logger.error(f"Error getting {agent_name} response: {e}")
            error_message = f"No response from {agent_name} due to error: {str(e)}"
            self.log_message(agent_name, error_message)
            return {"response": error_message, "content": error_message, "tool_calls": [], "tool_attempts": 0, "rag_used": False}
    
    async def generate_pitch(self, facts, product_description, product_key):
        """generate a pitch for a SharkTank product using asynchronous specialists with both RAG and DuckDuckGo search"""
        facts_str = str(facts) if not isinstance(facts, str) else facts
        product_description_str = str(product_description) if not isinstance(product_description, str) else product_description
        
        self.logger.info(f"Generating pitch for: {product_key}")
        
        total_input_length = len(product_description_str) + len(facts_str)
        total_output_length = 0
        
        self.logger.info(f"Initial input prompt length: {total_input_length}")
        
        # base context without tool instructions
        base_context = f"""
        FACTS:
        {facts_str}
        
        PRODUCT DESCRIPTION:
        {product_description_str}
        """
        
        # context with ddg instructions
        specialist_context = base_context + """
        INSTRUCTIONS:
        Analyze the provided facts and product description from your specialist perspective.
        Use the DuckDuckGo Search tool to find additional information that will enhance your analysis.
        """
        
        specialist_results = {}
        
        # track rate limiting across agents
        rate_limit_encountered = False
        tool_calls_count = 0
        
        start_time = time.time()
        
        self.logger.info("Starting all specialist agents concurrently with DuckDuckGo search enabled...")
        
        # create tasks for specialists using specialist context
        tasks = {
            "Financial Strategist": self.run_agent_async(
                self.financial_strategist, 
                "Financial Strategist", 
                specialist_context
            ),
            "Market Research Specialist": self.run_agent_async(
                self.market_research_specialist, 
                "Market Research Specialist", 
                specialist_context
            ),
            "Product Technical Advisor": self.run_agent_async(
                self.product_technical_advisor, 
                "Product Technical Advisor", 
                specialist_context
            ),
            "Shark Psychology Expert": self.run_agent_async(
                self.shark_psychology_expert, 
                "Shark Psychology Expert", 
                specialist_context
            )
        }
        
        # track progress
        specialist_progress = {name: "In progress" for name in tasks.keys()}
        specialist_contents = {}
        
        self.logger.info("Waiting for all specialists to complete concurrently...")
        
        # use helper dict to track tasks
        task_objs = list(tasks.values())
        task_names = list(tasks.keys())
        
        # wait for all tasks
        results = await asyncio.gather(*task_objs, return_exceptions=True)
        
        # process results
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                specialist_progress[name] = f"Error: {str(result)}"
                self.logger.error(f"{name}: Error - {str(result)}")
                continue
                
            specialist_results[name] = result
            
            if "content" in result:
                content = strip_thinking_tokens(result["content"])
                specialist_contents[name] = content
                specialist_progress[name] = "Completed"
                self.logger.info(f"{name}: Completed (Length: {len(content)})")
                total_output_length += len(content)
            else:
                specialist_progress[name] = "Error: No content"
                self.logger.warning(f"{name}: Error - No content")
            
            # count tool attempts
            if "tool_attempts" in result:
                agent_tool_calls = result["tool_attempts"]
                tool_calls_count += agent_tool_calls
                self.logger.info(f"{name} made {agent_tool_calls} tool attempts")
            
            # check for rate limits
            if "tool_calls" in result and result["tool_calls"]:
                for tool_call in result["tool_calls"]:
                    if hasattr(tool_call, 'error') and "rate limit" in str(tool_call.error).lower():
                        rate_limit_encountered = True
        
        self.logger.info("Specialist agents completion status:")
        for name, status in specialist_progress.items():
            self.logger.info(f"  {name}: {status}")
        
        self.logger.info(f"Total tool calls/attempts by specialists: {tool_calls_count}")
        if rate_limit_encountered:
            self.logger.warning("Rate limit errors were encountered during searches")
        
        # prepare drafter context without tool instructions
        drafter_context = base_context + "\n\n=== Suggestions from specialists ===\n\n"
        for name, content in specialist_contents.items():
            drafter_context += f"\n\n=== {name} ===\n\n{content}"
        
        drafter_context = strip_thinking_tokens(drafter_context)
        drafter_context_length = len(drafter_context)
        total_input_length += drafter_context_length
        
        self.logger.info("All specialists completed. Starting Pitch Drafter...")
        drafter_result = await self.run_agent_async(
            self.pitch_drafter, 
            "Pitch Drafter", 
            drafter_context
        )
        drafter_response = drafter_result["response"]
        drafter_content = drafter_result["content"]
        drafter_content = strip_thinking_tokens(drafter_content)
        
        drafter_response_length = len(drafter_content)
        total_output_length += drafter_response_length
        
        if hasattr(drafter_response, 'content') and isinstance(drafter_response.content, SharkTankPitch):
            draft_pitch = drafter_response.content
            self.logger.info("Successfully retrieved structured pitch from Pitch Drafter")
            draft_json = draft_pitch.model_dump()
        else:
            self.logger.error("Could not retrieve structured pitch from Pitch Drafter response. Using a placeholder.")
            draft_json = {
                "Pitch": "ERROR: Could not parse pitch from Drafter response.",
                "Initial_Offer": {
                    "Valuation": "Unknown",
                    "Equity_Offered": "Unknown",
                    "Funding_Amount": "Unknown",
                    "Key_Terms": "Unknown"
                }
            }
        
        # Prepare context for Pitch Critic - only includes base context and draft pitch
        critic_context = base_context + f"\n\n=== Draft Pitch ===\n\n{drafter_content}"
        critic_context = strip_thinking_tokens(critic_context)
        
        critic_context_length = len(critic_context)
        total_input_length += critic_context_length
        
        self.logger.info("Starting Pitch Critic with RAG...")
        critic_result = await self.run_agent_async(
            self.pitch_critic, 
            "Pitch Critic", 
            critic_context
        )
        
        critic_response = critic_result["response"]
        critic_content = critic_result["content"]
        critic_content = strip_thinking_tokens(critic_content)
        rag_used = critic_result.get("rag_used", False)
        
        critic_response_length = len(critic_content)
        total_output_length += critic_response_length
        
        # Prepare context for Pitch Finalizer - only includes base context, draft pitch, and critic feedback
        finalizer_context = base_context + f"\n\n=== Draft Pitch ===\n\n{drafter_content}"
        finalizer_context += f"\n\n=== Pitch Critic Feedback ===\n\n{critic_content}"
        finalizer_context = strip_thinking_tokens(finalizer_context)
        
        finalizer_context_length = len(finalizer_context)
        total_input_length += finalizer_context_length
        
        self.logger.info("Starting Pitch Finalizer...")
        finalizer_result = await self.run_agent_async(
            self.pitch_finalizer, 
            "Pitch Finalizer", 
            finalizer_context
        )
        
        finalizer_response = finalizer_result["response"]
        finalizer_content = finalizer_result["content"]
        finalizer_content = strip_thinking_tokens(finalizer_content)
        
        finalizer_response_length = len(finalizer_content)
        total_output_length += finalizer_response_length
        
        if hasattr(finalizer_response, 'content') and isinstance(finalizer_response.content, SharkTankPitch):
            final_pitch = finalizer_response.content
            self.logger.info("Successfully retrieved structured pitch from Pitch Finalizer")
            final_json = final_pitch.model_dump()
        else:
            self.logger.error("Could not retrieve structured pitch from Pitch Finalizer response. Using draft pitch.")
            final_json = draft_json
        
        execution_time = time.time() - start_time
        
        output_pitch_length = len(json.dumps(final_json, indent=2))
        total_output_length += output_pitch_length
        
        self.logger.info(f"Output pitch length: {output_pitch_length}")
        self.logger.info(f"Total input prompt length: {total_input_length}")
        self.logger.info(f"Total output response length: {total_output_length}")
        self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
        self.logger.info(f"Total tool calls/attempts: {tool_calls_count}")
        self.logger.info(f"RAG used by Pitch Critic: {rag_used}")
        
        # Save final pitch and metrics to output files
        with open(os.path.join(self.output_dir, "final_pitch.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)
        
        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "total_execution_time_seconds": execution_time,
                "total_input_prompt_length": total_input_length,
                "total_output_response_length": total_output_length,
                "total_tool_calls": tool_calls_count,
                "rag_used": rag_used,
                "specialist_status": specialist_progress,
                "agent_input_lengths": {
                    "specialists": len(specialist_context),
                    "pitch_drafter": drafter_context_length,
                    "pitch_critic": critic_context_length,
                    "pitch_finalizer": finalizer_context_length
                },
                "agent_output_lengths": {
                    **{name: len(content) for name, content in specialist_contents.items()},
                    "pitch_drafter": len(drafter_content),
                    "pitch_critic": len(critic_content),
                    "pitch_finalizer": len(finalizer_content)
                },
                "tool_usage": {
                    "ddg_enabled_for_specialists": True,
                    "ddg_tool_calls_count": tool_calls_count,
                    "rag_enabled_for_critic": True,
                    "rag_used_by_critic": rag_used
                }
            }, f, indent=2)
        
        return final_json 

def load_data(file_path: str) -> Any:
    """load data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return {}

def setup_arg_parser():
    """set up and return the argument parser for command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches using an asynchronous multi-agent system with both RAG and DuckDuckGo search tools.')
    
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model ID to use for all agents (default: gpt-4o)')
    
    parser.add_argument('--pdf-urls', type=str, nargs='+',
                       help='URLs of PDF documents to load into the knowledge base (default: HBS pitch materials)')
    
    parser.add_argument('--pdf-paths', type=str, nargs='+',
                       help='Local paths to PDF documents (default: data/pdfs/hbs_*.pdf)')
    
    parser.add_argument('--chunking', type=str, default='fixed', choices=['fixed', 'agentic', 'semantic'],
                       help='Chunking strategy for the knowledge base (default: fixed)')

    parser.add_argument('--force-reload', action='store_true',
                       help='Force reloading and re-embedding of documents even if they exist in the database')
    
    parser.add_argument('--skip-embedding-check', action='store_true',
                       help='Skip checking if documents are already embedded (useful if check is failing)')
    
    parser.add_argument('--product-key', type=str, default='facts_shark_tank_transcript_0_GarmaGuard.txt',
                        help='Product key/filename to process from the facts data (default: facts_shark_tank_transcript_0_GarmaGuard.txt)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--no-debug', action='store_false', dest='debug',
                        help='Disable debug mode for detailed agent logging (default: enabled)')
    
    return parser

async def main_async():
    """run the SharkTank pitch generator asynchronously with both RAG and DuckDuckGo search tools"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/async_specialist_team_rag_ddg_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_name = f"full_log_async_specialist_team_rag_ddg_{timestamp}.log"
    log_file_path = os.path.join(output_dir, log_file_name)
    
    # clean up existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger(f"async_pitch_generator_rag_ddg_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(log_file_path, 'a', 'utf-8', original_stdout)
    sys.stderr = TeeLogger(log_file_path, 'a', 'utf-8', original_stderr)
    
    logger.info(f"Starting Async Specialist Team with RAG and DuckDuckGo, model: {args.model}, product key: {args.product_key}, max_tokens: {args.max_tokens}, temperature: {args.temperature}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    # Set environment variables for RAG from command-line arguments
    if args.chunking:
        os.environ["ASYNC_RAG_CHUNKING_STRATEGY"] = args.chunking
    if args.force_reload:
        os.environ["ASYNC_RAG_FORCE_RELOAD"] = "true"
    if args.skip_embedding_check:
        os.environ["ASYNC_RAG_SKIP_EMBEDDING_CHECK"] = "true"
    
    # Try multiple possible locations for the data file
    facts_file_path = None
    possible_paths = [
        "./data/all_processed_facts.json",
        "./ganwang/all_processed_facts.json",
        os.path.join(os.path.dirname(__file__), "all_processed_facts.json")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            facts_file_path = path
            break
    
    if not facts_file_path:
        logger.error(f"Facts file not found. Tried: {possible_paths}")
        return
    
    all_facts = load_data(facts_file_path)
    if not all_facts:
        logger.error("Failed to load facts data")
        return
    
    if args.product_key not in all_facts:
        available_keys = list(all_facts.keys())
        logger.error(f"Product key '{args.product_key}' not found in facts data.")
        logger.info(f"Available keys: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}")
        return
    
    episode_key = args.product_key
    episode_data = all_facts[episode_key]
    
    facts = str(episode_data.get('facts', "")) if isinstance(episode_data, dict) else str(episode_data)
    product_description = str(episode_data.get('product_description', "")) if isinstance(episode_data, dict) else str(episode_data)
    
    logger.info(f"Processing data for product file: {episode_key}")
    
    # save input data for later reference
    with open(os.path.join(output_dir, "input_data.json"), "w", encoding="utf-8") as f:
        json.dump({
            "episode_key": episode_key,
            "facts": facts,
            "product_description": product_description,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "chunking_strategy": args.chunking
        }, f, indent=2)
    
    team = AsyncSpecialistTeamRAGDDG(
        output_dir=output_dir, 
        log_file_path=log_file_path, 
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug
    )
    
    try:
        final_pitch = await team.generate_pitch(facts, product_description, episode_key)
        logger.info(f"Final pitch generated and saved to {output_dir}/final_pitch.json")
    except Exception as e:
        logger.error(f"Error generating pitch: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return

def main():
    """entry point that calls the async main function"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async())

if __name__ == "__main__":
    main() 