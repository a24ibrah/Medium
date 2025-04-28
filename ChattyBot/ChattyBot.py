import streamlit as st
import os
import requests # Keep for sync fallback or specific needs if any
from urllib.parse import urlparse, urljoin
import re
import shutil
import tempfile
from collections import defaultdict
import asyncio # Import asyncio
import aiohttp # Import aiohttp for async requests

# Add dotenv to load environment variables
from dotenv import load_dotenv

# Libraries for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
# Removed unused import: from langchain.schema.runnable import RunnableConfig


# Libraries for Web Search/Crawl
from duckduckgo_search import DDGS
import trafilatura # Still useful for extraction logic
from bs4 import BeautifulSoup # For parsing robots.txt (simplified)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get API key and Base URL from environment variables
# Set them as standard OpenAI environment variables for ChatOpenAI to pick up
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" # This remains constant for OpenRouter

# Set standard OpenAI environment variables that ChatOpenAI reads
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = OPENROUTER_BASE_URL # Use OPENAI_API_BASE or OPENAI_BASE_URL

# Choose a suitable model from OpenRouter
# Keep the model name as specified in the last code block you shared
MODEL_NAME = "mistralai/mistral-7b-instruct:free"

# OpenRouter-specific HTTP headers - these are often used by OpenRouter themselves
# or might need to be passed via specific config, but not directly to create() apparently.
# We will define them but won't pass them to ChatOpenAI constructor explicitly
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://myapp.example.com",  # Replace with your app's URL
    "X-Title": "Chatty Web Search App"            # Replace with your app name
}

# Print a debug message if the API key is empty or not set
if not OPENROUTER_API_KEY:
    print("WARNING: OpenRouter API key is empty or not set in .env file.")
    # In Streamlit, you can also display this error immediately
    st.error("OpenRouter API key is not set. Please check your `.env` file or environment variables.")
    st.stop()


# RAG Parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
TOP_K_RETRIEVAL = 6

# Web Search Parameters
MAX_SEARCH_RESULTS = 8
MAX_PAGES_TO_CRAWL = 5 # Limit the number of pages we actually attempt to crawl concurrently
CRAWL_TIMEOUT = 10 # seconds for fetching a single page


# --- Helper Functions (ASYNC) ---
# (These functions remain the same as the async version)

def simplify_url_to_tag(url):
    """Simplifies a URL to a short tag for citation."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split('.')
        if len(domain) >= 2:
            tag_parts = [d for d in domain if d not in ['www', 'co', 'uk', 'com', 'org', 'net']]
            if tag_parts:
                tag = '_'.join(tag_parts[-2:])
            else:
                tag = domain[0]
        else:
            tag = parsed_url.netloc.replace('.', '_') or "site"

        tag = tag.replace('-', '_')
        if not tag or tag in ['com', 'org', 'net', 'co', 'uk']:
            tag = 'site'
        return tag.lower()[:20]
    except Exception:
        return "site"


async def async_fetch(session, url, timeout=CRAWL_TIMEOUT):
    """Asynchronously fetches content from a URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ethical-search-app/1.0)'}
    try:
        async with session.get(url, timeout=timeout, headers=headers) as response:
            response.raise_for_status()
            return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    except Exception as e:
        # Catch unexpected errors during fetch
        # print(f"Unexpected error fetching {url}: {e}") # Use print instead of st.warning in async functions
        return None


async def async_is_url_allowed_by_robots(session, url, user_agent="*"):
    """Asynchronously checks robots.txt Disallow rules."""
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")
        path = parsed_url.path or '/'
        if parsed_url.query:
            path += '?' + parsed_url.query

        robots_content = await async_fetch(session, robots_url, timeout=5)
        if robots_content is None:
            return True, url

        disallowed_paths = []
        current_agent = None
        for line in robots_content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                if key.lower() == 'user-agent':
                    current_agent = value
                elif key.lower() == 'disallow':
                    if current_agent is None or current_agent == user_agent or current_agent == '*':
                        disallowed_paths.append(value)

        for disallowed_path in disallowed_paths:
            if disallowed_path and path.startswith(disallowed_path):
                # print(f"Disallowed by robots.txt: {url}") # Use print
                return False, url

        return True, url

    except Exception:
        # print(f"Error checking robots.txt for {url}: {e}") # Use print
        return True, url


async def async_crawl_and_extract(session, url):
    """Asynchronously fetches URL content and extracts main text."""
    try:
        html_content = await async_fetch(session, url)

        if html_content:
            markdown_content = trafilatura.extract(html_content, output_format='markdown', include_links=False, include_images=False, include_tables=False)
            if markdown_content and len(markdown_content) > 100:
                tag = simplify_url_to_tag(url)
                return Document(page_content=markdown_content, metadata={"source": tag, "url": url})
        return None
    except Exception:
        # print(f"Could not crawl or extract content from {url}: {e}") # Use print
        return None


async def process_urls_concurrently(urls):
    """Performs robots.txt checks and crawling concurrently."""
    allowed_urls = []
    crawled_documents = []

    async with aiohttp.ClientSession() as session:
        st.info("Checking robots.txt concurrently...")
        robots_check_tasks = [async_is_url_allowed_by_robots(session, url) for url in urls]
        results = await asyncio.gather(*robots_check_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            is_allowed, url = result
            if is_allowed:
                allowed_urls.append(url)

        st.info(f"Found {len(allowed_urls)} URLs allowed by robots.txt (out of {len(urls)} searched).")

        urls_to_crawl = allowed_urls[:MAX_PAGES_TO_CRAWL]
        if not urls_to_crawl:
            st.warning("No URLs allowed by robots.txt or exceeding crawl limit to crawl.")
            return []

        st.info(f"Crawling up to {len(urls_to_crawl)} allowed pages concurrently...")
        crawl_tasks = [async_crawl_and_extract(session, url) for url in urls_to_crawl]
        crawled_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)

        for doc_or_exception in crawled_results:
            if isinstance(doc_or_exception, Document):
                crawled_documents.append(doc_or_exception)
            elif isinstance(doc_or_exception, Exception):
                pass

        st.info(f"Successfully crawled and extracted content from {len(crawled_documents)} pages.")

    return crawled_documents


def build_system_prompt(with_context):
    """Constructs the system prompt based on the mode."""
    base_prompt = """You are a helpful, neutral, and concise AI assistant.

Strictly follow these constraints and instructions:
- Your total response length must be concise and ideally under 300 words.
- Do not include large quotations; summarize information instead.
- Do not include full URLs in your response. Use simplified source tags.
- Refuse any request for disallowed, harmful, or unethical content.
- Format your final output clearly with a concise answer first, then cited bullet points, and finally the Sources line.
"""

    if with_context:
        context_prompt = """
You are in Web-search-augmented mode.
- Use ONLY the information provided in the CONTEXT START / CONTEXT END block to answer the USER QUESTION.
- If the provided context is insufficient to fully answer the question, state clearly: "I couldn't find that in the provided sources." Do NOT use your internal knowledge if context is provided and insufficient.
- After every sentence or phrase in your concise answer and after every bullet point that uses information from the context, append one or more source tags [tag][anothertag] corresponding to the source snippets used. If multiple snippets support a piece of information, list all relevant tags.
- At the very end of your response, after the bullet points, list all unique source tags used throughout your answer in the format "Sources: tag1, tag2, tag3, ...". Do not include a Sources line if no context was available or used.
- Your answer should include:
    1. A concise summary answer.
    2. Up to five cited bullet points supporting the answer.
    3. The final "Sources:" line listing all used tags.
"""
        return base_prompt + context_prompt
    else:
        direct_prompt = """
You are in Direct-LLM mode.
- Answer the USER QUESTION using your vast internal knowledge.
- There is NO CONTEXT block provided in this mode. Ignore any CONTEXT START / CONTEXT END markers and `with_context: false` markers.
- Since your knowledge has a cutoff date, the information might be outdated. Include a brief warning at the end of your response stating that the information may not be current.
- Do NOT include any source tags or a "Sources:" line, as you are not using external documents.
- Your answer should include:
    1. A concise summary answer.
    2. Up to five relevant bullet points from your knowledge.
    3. A final line indicating the knowledge cutoff.
"""
        return base_prompt + direct_prompt

# --- Streamlit App ---

st.set_page_config(page_title="LLM with RAG and Web Search", page_icon="🔍")

st.title("🌐 LLM with Web Search & RAG")
st.markdown("Choose your mode and ask a question.")

mode = st.radio(
    "Choose Mode:",
    ("Direct LLM", "Web Search Augmented"),
    help="Direct LLM uses the model's internal knowledge (may be outdated). Web Search mode finds current info online and uses it to answer."
)

user_query = st.text_input("Your Question:", placeholder="e.g., Who won the Grammys for Best Rap Album in 2025?")

if st.button("Ask"):
    if not user_query:
        st.warning("Please enter a question.")
        st.stop()

    st.markdown("---")
    response_placeholder = st.empty() # Placeholder for streaming response
    full_response = ""
    temp_chroma_dir = None # Variable to hold the temporary directory path
    db = None # Initialize db for cleanup

    # Determine the initial mode based on user selection
    current_mode_is_rag = (mode == "Web Search Augmented")
    context = ""
    sources_tags = set() # Set to store unique source tags found in retrieved docs
    retrieved_docs = [] # Initialize retrieved_docs list

    # The check now happens earlier based on os.environ.get, handled by the st.error above

    if current_mode_is_rag:
        st.info("Running in Web Search Augmented mode...")
        crawled_documents = [] # Store results from async crawl

        try:
            with st.spinner("Searching the web..."):
                # Use DDGS text search (synchronous call)
                search_results = DDGS().text(user_query, max_results=MAX_SEARCH_RESULTS, safesearch='moderate') # Removed lang='en'
                urls = [result['href'] for result in search_results if 'href' in result and result['href']]
                st.markdown(f"Found {len(urls)} potential URLs.")

            if not urls:
                 st.warning("No search results found.")
                 st.info("Proceeding in Direct LLM mode.")
                 current_mode_is_rag = False
                 pass # Skip crawling/RAG


            if current_mode_is_rag and urls:
                # --- Run the asynchronous crawling process ---
                crawled_documents = asyncio.run(process_urls_concurrently(urls))

                if not crawled_documents:
                    st.warning("Couldn't crawl or extract content from any pages.")
                    st.info("Proceeding in Direct LLM mode due to lack of context.")
                    current_mode_is_rag = False # Switch mode
                    context = "[No context found from search/crawl]"
                    pass # Continue execution flow


            if current_mode_is_rag and crawled_documents: # Only proceed with RAG if docs were crawled
                with st.spinner(f"Chunking and embedding {len(crawled_documents)} documents..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    chunks = text_splitter.split_documents(crawled_documents)

                    if not chunks:
                         st.warning("Document chunking failed.")
                         st.info("Proceeding in Direct LLM mode due to lack of usable chunks.")
                         current_mode_is_rag = False # Switch mode
                         context = "[No usable chunks generated]"
                         pass # Continue execution flow
                    else:
                        # Create a temporary directory for Chroma persistence
                        temp_chroma_dir = tempfile.mkdtemp()

                        embedding_model = "nomic-ai/nomic-embed-text-v1.5" # Keep the embedding model
                        try:
                            # Initialize embeddings - should pick up key and base from environment variables
                            embeddings = OpenAIEmbeddings(
                                # No api_key or base_url needed here if set in env vars
                                model=embedding_model
                            )
                            # Perform a quick test embedding to catch model errors early
                            _ = embeddings.embed_documents(["test document"])

                        except Exception as e:
                            st.error(f"Error initializing embedding model '{embedding_model}': {e}")
                            st.info("Proceeding in Direct LLM mode as embedding failed.")
                            current_mode_is_rag = False
                            context = f"[Error with embedding model: {e}]"
                            # Cleanup potentially created temp dir before stopping RAG
                            if temp_chroma_dir and os.path.exists(temp_chroma_dir):
                                shutil.rmtree(temp_chroma_dir)
                                temp_chroma_dir = None
                            pass # Skip RAG steps

                        if current_mode_is_rag: # Check mode again
                            db = Chroma.from_documents(
                                chunks,
                                embeddings,
                                persist_directory=temp_chroma_dir
                            )

                            retriever = db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})

                if current_mode_is_rag: # Check mode again
                    with st.spinner("Retrieving relevant information..."):
                         retrieved_docs = retriever.invoke(user_query)

                    if not retrieved_docs:
                         st.warning("Vector store retrieval failed to find relevant chunks.")
                         st.info("Proceeding in Direct LLM mode due to lack of relevant context.")
                         current_mode_is_rag = False
                         context = "[No relevant chunks retrieved]"
                         pass
                    else:
                        context_parts = []
                        seen_content = set()
                        for doc in retrieved_docs:
                            if doc.page_content not in seen_content:
                                tag = doc.metadata.get("source", "unknown")
                                context_parts.append(f"[{tag}] {doc.page_content}")
                                sources_tags.add(tag)
                                seen_content.add(doc.page_content)

                        context = "\n\n---\n\n".join(context_parts)
                        st.markdown("Context built from sources.")


        except Exception as e:
            st.error(f"An unexpected error occurred during web search/RAG setup: {e}")
            st.info("Attempting to proceed in Direct LLM mode.")
            current_mode_is_rag = False
            context = f"[Error during context setup: {e}]"


    # --- LLM Interaction ---

    system_prompt = build_system_prompt(current_mode_is_rag)

    user_message_content = f"""USER QUESTION:   {user_query}

CONTEXT START
{context}
CONTEXT END

with_context: {current_mode_is_rag}
"""

    try:
        # Create the LLM
        # It should pick up the API key and base URL from the environment variables
        llm = ChatOpenAI(
            # No base_url or api_key parameters needed here if set in env vars
            model=MODEL_NAME,
            streaming=True,
            temperature=0.1,
            max_tokens=400
        )

        st.info(f"Using model: {MODEL_NAME} via OpenRouter")
        st.info(f"Generating response in {'Web Search Augmented' if current_mode_is_rag else 'Direct LLM'} mode...")

        for chunk in llm.stream([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message_content}]):
            delta = chunk.content or ""
            full_response += delta
            response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)

        response_placeholder.markdown(full_response, unsafe_allow_html=True)

        if not current_mode_is_rag:
             st.info("Note: Answer generated from model knowledge (may be outdated). No external sources cited.")


    except Exception as e:
        st.error(f"An error occurred during LLM generation: {e}")
        # The API key and model checks should now happen based on env vars picked up by ChatOpenAI
        st.error(f"An unexpected error occurred with the LLM: {e}")


    finally:
        # --- Cleanup ---
        if temp_chroma_dir and os.path.exists(temp_chroma_dir):
            st.info(f"Cleaning up temporary Chroma directory: {temp_chroma_dir}")
            try:
                if 'db' in locals() and db is not None:
                     del db
                shutil.rmtree(temp_chroma_dir)
            except PermissionError as e:
                st.error(f"Permission error during cleanup of {temp_chroma_dir}. Directory might still be in use: {e}")
                st.info("You may need to manually delete the temporary directory after the app closes.")
            except Exception as e:
                st.error(f"Error during cleanup of {temp_chroma_dir}: {e}")