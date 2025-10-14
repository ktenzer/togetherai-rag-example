#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal, robust website crawler + Together embeddings → Chroma DB ingester.

Enhanced for any type of website with multiple fallback strategies:
- JavaScript rendering support (Selenium + Playwright)
- Multiple sitemap discovery methods
- Anti-bot evasion techniques
- Content extraction for different site types
- Duplicate detection and URL normalization
- Configurable crawling strategies

Usage:
  python ingest_robust_web.py --url https://example.com --max-pages 100 --chroma ./chroma_db

Features:
- Automatically detects if site needs JavaScript rendering
- Tries multiple User-Agents and headers
- Discovers sitemaps from multiple sources
- Extracts content using readability algorithms
- Handles rate limiting and retries intelligently
- Works with blogs, e-commerce, documentation, news sites, etc.

Env:
  TOGETHER_API_KEY must be set (dotenv supported)
"""

import os, sys, time, textwrap, warnings, logging, json, platform, re, hashlib, random
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin, urldefrag, quote
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib import robotparser
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional JavaScript rendering
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

import numpy as np
import torch, chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from dotenv import load_dotenv
from transformers import AutoTokenizer
from together import Together
from tqdm import tqdm

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

# ------------------ Env / Logging ------------------
load_dotenv(override=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

CHROMA_DIR = Path("./chroma_db")

# ------------------ Models / Config ----------------
TEXT_MODEL     = "BAAI/bge-base-en-v1.5"   # Together embeddings (512-token limit)
MAX_EMB_TOKENS = 480
CHUNK_OVERLAP  = 32

# Crawling configuration
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_SLEEP_BETWEEN_REQUESTS = 1.0
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 2.0
JS_WAIT_TIME = 3  # seconds to wait for JS to render
MAX_REDIRECT_DEPTH = 5

# Content filtering
MIN_CONTENT_LENGTH = 100
MAX_CONTENT_LENGTH = 50000

# Common sitemap locations to try
SITEMAP_PATHS = [
    '/sitemap.xml',
    '/sitemap_index.xml', 
    '/sitemaps.xml',
    '/sitemap/sitemap-index.xml',
    '/sitemap/index.xml',
    '/wp-sitemap.xml',  # WordPress
    '/sitemap-index.xml',
    '/site_map.xml'
]

# Optimize for MacOS / GPU
IS_MAC = platform.system() == "Darwin"
if torch.cuda.is_available():
    device = "cuda"
elif IS_MAC and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
else:
    device = "cpu"

try:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
except Exception:
    pass

# Together client (uses TOGETHER_API_KEY)
together_client = Together()

# Tokenizer (for counting/splitting tokens only)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)

# User agent rotation
ua = UserAgent()

# ------------------ Enhanced Data Classes ------------------
@dataclass(frozen=True)
class CrawlConfig:
    """Configuration for web crawling behavior."""
    max_pages: int = 1000
    max_depth: int = 3
    restrict_to_domain: bool = True
    restrict_to_path: bool = False
    use_javascript: bool = True  # Try JS rendering if regular fails
    respect_robots_txt: bool = True
    enable_sitemap_discovery: bool = True
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    sleep_between_requests: float = DEFAULT_SLEEP_BETWEEN_REQUESTS
    retry_count: int = DEFAULT_RETRY_COUNT
    retry_delay: float = DEFAULT_RETRY_DELAY
    user_agent_rotation: bool = True
    random_delays: bool = True

@dataclass
class CrawlStats:
    """Track crawling statistics."""
    urls_discovered: int = 0
    urls_processed: int = 0
    pages_successful: int = 0
    pages_failed: int = 0
    js_rendered_pages: int = 0
    duplicate_urls: int = 0
    
@dataclass(frozen=True)
class Page:
    url: str
    title: str
    content: str
    content_type: str = "html"  # html, markdown, text
    meta_description: str = ""
    language: str = "en"
    content_hash: str = field(init=False)
    
    def __post_init__(self):
        # Generate content hash for duplicate detection
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        object.__setattr__(self, 'content_hash', content_hash)

# ------------------ Embedding helpers (same as before) ------------------
class SciEmbedding(EmbeddingFunction):
    """Schema-only embedding fn that matches inference."""
    def __init__(self):
        self.dim = 768  # bge-base-en-v1.5
    def name(self): return TEXT_MODEL
    def dimensions(self): return self.dim
    def __call__(self, texts):
        return embed_texts_token_safe_batch(list(texts))

def _text_to_ids(text: str):
    return tokenizer.encode(text, add_special_tokens=False)

def _ids_to_text(ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def _split_ids(ids: List[int], max_len=MAX_EMB_TOKENS, overlap=CHUNK_OVERLAP):
    """Yield max_len windows over token ids with fixed overlap."""
    if len(ids) <= max_len:
        yield ids
        return
    step = max_len - overlap
    for start in range(0, len(ids), step):
        window = ids[start:start + max_len]
        if not window:
            break
        yield window

def _embed_many_texts(texts: List[str]) -> List[List[float]]:
    """Single Together embeddings call for a list of texts (each ≤512 tokens)."""
    if not texts:
        return []
    resp = together_client.embeddings.create(model=TEXT_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_text_token_safe(text: str) -> List[float]:
    """Token-safe: split >512-token text → embed windows → mean-pool to one vector."""
    ids = _text_to_ids(text)
    chunks = list(_split_ids(ids))
    if len(chunks) == 1:
        return _embed_many_texts([text])[0]
    chunk_texts = [_ids_to_text(c) for c in chunks]
    chunk_vecs = _embed_many_texts(chunk_texts)
    return (np.mean(np.array(chunk_vecs, dtype=np.float32), axis=0)).tolist()

def embed_texts_token_safe_batch(texts: List[str], chunk_batch_size: int = 96) -> List[List[float]]:
    """
    Faster, batched embeddings via Together.ai with token-safe chunking.
    """
    # tokenize & split each input into chunk strings
    chunk_texts: List[str] = []
    seg_sizes: List[int] = []
    for t in texts:
        ids = _text_to_ids(t)
        chunks = list(_split_ids(ids))
        seg_sizes.append(len(chunks))
        chunk_texts.extend([_ids_to_text(c) for c in chunks])

    if not chunk_texts:
        return []

    # embed all chunks in batches
    chunk_vecs: List[List[float]] = []
    for i in range(0, len(chunk_texts), chunk_batch_size):
        sub = chunk_texts[i:i + chunk_batch_size]
        chunk_vecs.extend(_embed_many_texts(sub))

    # mean-pool back to per-document vectors
    out: List[List[float]] = []
    cursor = 0
    for n in seg_sizes:
        vecs = chunk_vecs[cursor:cursor + n]
        cursor += n
        out.append((np.mean(np.array(vecs, dtype=np.float32), axis=0)).tolist())
    return out

# ------------------ Enhanced URL helpers ------------------
def normalize_url(base: str, href: str) -> Optional[str]:
    """Normalize and clean URLs."""
    if not href:
        return None
    href = href.strip()
    
    # Skip various non-web protocols and fragments
    skip_schemes = ("mailto:", "tel:", "javascript:", "data:", "ftp:", "file:", "skype:")
    if any(href.startswith(scheme) for scheme in skip_schemes):
        return None
    
    try:
        abs_url = urljoin(base, href)
        abs_url, _frag = urldefrag(abs_url)  # drop #fragment
        parsed = urlparse(abs_url)
        
        if parsed.scheme not in {"http", "https"}:
            return None
            
        # Basic URL cleaning
        abs_url = abs_url.rstrip('/')
        
        # Skip common non-content URLs
        skip_patterns = [
            r'/wp-admin/', r'/admin/', r'/login', r'/logout', r'/register',
            r'\.(css|js|json|xml|rss|atom|pdf|doc|docx|xls|xlsx|ppt|pptx)$',
            r'\.(jpg|jpeg|png|gif|bmp|svg|ico|webp)$',
            r'\.(mp3|mp4|avi|mov|wmv|flv|wav)$',
            r'\.(zip|rar|tar|gz|7z)$'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, abs_url, re.IGNORECASE):
                return None
                
        return abs_url
    except Exception:
        return None

def same_domain(u1: str, u2: str) -> bool:
    """Check if URLs are from the same domain."""
    try:
        domain1 = urlparse(u1).netloc.lower()
        domain2 = urlparse(u2).netloc.lower()
        # Handle subdomains - consider www.example.com and example.com as same domain
        domain1 = domain1.replace('www.', '')
        domain2 = domain2.replace('www.', '')
        return domain1 == domain2
    except:
        return False

def under_path(u: str, root: str) -> bool:
    """Check if URL is under root path."""
    try:
        p, r = urlparse(u), urlparse(root)
        return p.path.startswith(r.path)
    except:
        return False

# ------------------ Enhanced Session Management ------------------
def create_robust_session(config: CrawlConfig) -> requests.Session:
    """Create a robust session with retries and realistic headers."""
    session = requests.Session()
    
    # Setup retry strategy
    retry_strategy = Retry(
        total=config.retry_count,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set realistic headers
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0"
    }
    
    if config.user_agent_rotation:
        headers["User-Agent"] = ua.random
    else:
        headers["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    session.headers.update(headers)
    return session

# ------------------ JavaScript Rendering ------------------
class JavaScriptRenderer:
    """Handles JavaScript rendering using Selenium or Playwright."""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.driver = None
        self.playwright_context = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up browser resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            
        if self.playwright_context:
            try:
                self.playwright_context.close()
            except:
                pass
            self.playwright_context = None
    
    def render_with_selenium(self, url: str) -> Optional[str]:
        """Render page using Selenium Chrome."""
        if not SELENIUM_AVAILABLE:
            return None
            
        try:
            if not self.driver:
                options = ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument(f'--user-agent={ua.random}')
                
                self.driver = webdriver.Chrome(options=options)
                self.driver.set_page_load_timeout(self.config.request_timeout)
                
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, JS_WAIT_TIME).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            return self.driver.page_source
            
        except Exception as e:
            print(f"Selenium rendering failed for {url}: {e}")
            return None
    
    def render_with_playwright(self, url: str) -> Optional[str]:
        """Render page using Playwright."""
        if not PLAYWRIGHT_AVAILABLE:
            return None
            
        try:
            if not self.playwright_context:
                from playwright.sync_api import sync_playwright
                self.playwright = sync_playwright().start()
                browser = self.playwright.chromium.launch(headless=True)
                self.playwright_context = browser.new_context(
                    user_agent=ua.random,
                    viewport={'width': 1920, 'height': 1080}
                )
                
            page = self.playwright_context.new_page()
            page.goto(url, timeout=self.config.request_timeout * 1000, wait_until='networkidle')
            content = page.content()
            page.close()
            return content
            
        except Exception as e:
            print(f"Playwright rendering failed for {url}: {e}")
            return None
    
    def render_page(self, url: str) -> Optional[str]:
        """Try to render page with JS, fallback between methods."""
        # Try Playwright first (usually faster and more reliable)
        content = self.render_with_playwright(url)
        if content:
            return content
            
        # Fallback to Selenium
        content = self.render_with_selenium(url)
        if content:
            return content
            
        return None

# ------------------ Enhanced Content Extraction ------------------
def extract_main_content(soup: BeautifulSoup, url: str = "") -> Tuple[str, str, str]:
    """
    Enhanced content extraction that works well with different site types.
    Returns (title, content, meta_description)
    """
    # Get title
    title_tag = soup.find("title")
    title = (title_tag.text or "").strip() if title_tag else ""
    
    # Get meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()
    
    # Remove unwanted elements
    for element in soup(["script", "style", "noscript", "iframe", "svg", "nav", 
                        "footer", "header", "aside", "advertisement", ".ad"]):
        element.decompose()
    
    # Try to find main content area using common patterns
    main_content = None
    
    # Priority selectors for main content
    main_selectors = [
        "main", "article", ".main-content", "#main-content", "#content", 
        ".content", ".post-content", ".entry-content", ".page-content",
        ".article-content", ".blog-content", "[role='main']", ".documentation",
        ".docs-content", ".wiki-content"
    ]
    
    for selector in main_selectors:
        elements = soup.select(selector)
        if elements:
            main_content = elements[0]
            break
    
    # If no main content found, use body but remove common sidebars/navigation
    if not main_content:
        main_content = soup.find("body")
        if main_content:
            # Remove sidebars, navigation, and other non-content elements
            for element in main_content(["nav", "sidebar", ".sidebar", "#sidebar",
                                       ".navigation", ".menu", ".breadcrumb",
                                       ".social", ".share", ".comments"]):
                element.decompose()
    
    if not main_content:
        return title, "", meta_desc
    
    # Convert to markdown-like text
    content = html_to_markdown_enhanced(main_content)
    
    return title, content, meta_desc

def html_to_markdown_enhanced(soup: BeautifulSoup) -> str:
    """
    Enhanced HTML to markdown conversion with better handling of different content types.
    """
    # Handle code blocks first
    for pre in soup.find_all("pre"):
        code = pre.get_text("\n", strip=False)
        lang = ""
        code_elem = pre.find("code")
        if code_elem and code_elem.get("class"):
            classes = code_elem.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
                    break
        pre.replace_with(f"\n```{lang}\n{code.rstrip()}\n```\n")
    
    # Handle inline code
    for code in soup.find_all("code"):
        if not code.find_parent("pre"):  # Skip if already in pre block
            code.replace_with(f"`{code.get_text()}`")
    
    # Handle lists
    for ul in soup.find_all("ul"):
        items = []
        for li in ul.find_all("li", recursive=False):
            text = " ".join(li.get_text().split())
            if text:
                items.append(f"- {text}")
        if items:
            ul.replace_with("\n" + "\n".join(items) + "\n")
    
    for ol in soup.find_all("ol"):
        items = []
        for i, li in enumerate(ol.find_all("li", recursive=False), 1):
            text = " ".join(li.get_text().split())
            if text:
                items.append(f"{i}. {text}")
        if items:
            ol.replace_with("\n" + "\n".join(items) + "\n")
    
    # Handle headings
    heading_tags = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}
    for tag_name, mark in heading_tags.items():
        for h in soup.find_all(tag_name):
            text = " ".join(h.get_text().split())
            if text:
                h.replace_with(f"\n{mark} {text}\n")
    
    # Handle links
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        text = " ".join(a.get_text().split())
        if text and href and not href.startswith("#"):
            a.replace_with(f"[{text}]({href})")
        else:
            a.replace_with(text)
    
    # Handle emphasis
    for strong in soup.find_all(["strong", "b"]):
        text = strong.get_text()
        if text:
            strong.replace_with(f"**{text}**")
    
    for em in soup.find_all(["em", "i"]):
        text = em.get_text()
        if text:
            em.replace_with(f"*{text}*")
    
    # Handle paragraphs and line breaks
    for br in soup.find_all("br"):
        br.replace_with("\n")
    
    for p in soup.find_all("p"):
        text = " ".join(p.get_text().split())
        if text:
            p.replace_with(f"{text}\n\n")
    
    # Handle tables
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["th", "td"]):
                cell_text = " ".join(td.get_text().split())
                cells.append(cell_text or " ")
            if cells:
                rows.append(cells)
        
        if rows:
            # Create markdown table
            if len(rows) >= 1:
                header = rows[0]
                md_table = "| " + " | ".join(header) + " |\n"
                md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
                
                for row in rows[1:]:
                    # Pad row to match header length
                    while len(row) < len(header):
                        row.append(" ")
                    md_table += "| " + " | ".join(row[:len(header)]) + " |\n"
                
                table.replace_with(f"\n{md_table}\n")
    
    # Handle blockquotes
    for quote in soup.find_all("blockquote"):
        lines = quote.get_text().strip().split('\n')
        quoted_lines = [f"> {line}" for line in lines if line.strip()]
        quote.replace_with("\n" + "\n".join(quoted_lines) + "\n")
    
    # Get final text
    text = soup.get_text("\n", strip=True)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

# ------------------ Sitemap Discovery ------------------
def discover_sitemaps(base_url: str, session: requests.Session, robots_content: str = None) -> List[str]:
    """
    Discover sitemaps using multiple strategies.
    """
    sitemaps = set()
    
    # 1. Extract from robots.txt
    if robots_content:
        for line in robots_content.split('\n'):
            line = line.strip()
            if line.lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                sitemaps.add(sitemap_url)
    
    # 2. Try common sitemap paths
    base_domain = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    for path in SITEMAP_PATHS:
        sitemap_url = urljoin(base_domain, path)
        try:
            resp = session.head(sitemap_url, timeout=10)
            if resp.status_code == 200:
                sitemaps.add(sitemap_url)
        except:
            pass
    
    # 3. Look for sitemap links in robots.txt location even if robots doesn't exist
    try:
        robots_url = urljoin(base_domain, '/robots.txt')
        resp = session.get(robots_url, timeout=10)
        if resp.status_code == 200:
            for line in resp.text.split('\n'):
                if 'sitemap' in line.lower() and 'http' in line:
                    # Extract URL from line
                    import re
                    urls = re.findall(r'https?://[^\s<>"]+', line)
                    for url in urls:
                        if 'sitemap' in url.lower():
                            sitemaps.add(url)
    except:
        pass
    
    return list(sitemaps)

def parse_sitemap(sitemap_url: str, session: requests.Session) -> List[str]:
    """
    Parse XML sitemap and extract URLs, handling sitemap indices.
    """
    urls = []
    try:
        resp = session.get(sitemap_url, timeout=DEFAULT_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return urls
        
        # Parse XML
        root = ET.fromstring(resp.text)
        
        # Handle namespace variations
        namespaces = {
            '': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }
        
        # Check if this is a sitemap index
        sitemap_elements = root.findall('.//sitemap', namespaces) or root.findall('.//sitemap')
        if sitemap_elements:
            # This is a sitemap index, parse child sitemaps
            for sitemap_elem in sitemap_elements[:5]:  # Limit to first 5 sitemaps
                loc_elem = sitemap_elem.find('loc', namespaces) or sitemap_elem.find('loc')
                if loc_elem is not None and loc_elem.text:
                    child_urls = parse_sitemap(loc_elem.text.strip(), session)
                    urls.extend(child_urls)
        else:
            # This is a regular sitemap, extract URLs
            url_elements = root.findall('.//loc', namespaces) or root.findall('.//loc')
            for url_elem in url_elements:
                if url_elem.text:
                    urls.append(url_elem.text.strip())
        
        return urls
        
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
        return urls

# ------------------ Robots.txt Handling ------------------
def is_allowed_by_robots(url: str, rp: robotparser.RobotFileParser, config: CrawlConfig) -> bool:
    """Enhanced robots.txt checking with fallbacks."""
    if not config.respect_robots_txt:
        return True
        
    try:
        # Check if robots.txt has any actual disallow rules
        has_disallow_rules = False
        for entry in rp.entries:
            if entry.rulelines:
                for rule in entry.rulelines:
                    if hasattr(rule, 'path') and rule.path and rule.allowance == False:
                        has_disallow_rules = True
                        break
        
        # If no explicit disallow rules found, allow everything
        if not has_disallow_rules:
            return True
        
        # Check with current user agent
        current_ua = config.user_agent_rotation if hasattr(config, 'current_ua') else "Mozilla/5.0"
        if rp.can_fetch(current_ua, url):
            return True
            
        # Try with generic user agent
        if rp.can_fetch('*', url):
            return True
            
        return False
        
    except Exception as e:
        # Be lenient if robots.txt parsing fails
        return True

# ------------------ Main Crawler Class ------------------
class RobustWebCrawler:
    """Main crawler class with multiple fallback strategies."""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = create_robust_session(config)
        self.stats = CrawlStats()
        self.seen_urls: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
        self.js_renderer = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        if self.js_renderer:
            self.js_renderer.cleanup()
        if self.session:
            self.session.close()
    
    def fetch_page(self, url: str, use_js: bool = False) -> Optional[requests.Response]:
        """Fetch a single page with fallbacks."""
        if use_js and self.config.use_javascript:
            if not self.js_renderer:
                self.js_renderer = JavaScriptRenderer(self.config)
            
            html_content = self.js_renderer.render_page(url)
            if html_content:
                # Create a fake response object
                resp = requests.Response()
                resp._content = html_content.encode('utf-8')
                resp.status_code = 200
                resp.headers['content-type'] = 'text/html'
                resp.url = url
                self.stats.js_rendered_pages += 1
                return resp
        
        # Regular HTTP request
        try:
            # Rotate user agent if enabled
            if self.config.user_agent_rotation:
                self.session.headers.update({"User-Agent": ua.random})
            
            resp = self.session.get(url, timeout=self.config.request_timeout)
            
            if resp.status_code == 429:  # Rate limited
                print(f"Rate limited at {url}, waiting...")
                time.sleep(self.config.retry_delay * 2)
                return None
                
            if resp.status_code >= 400:
                print(f"HTTP {resp.status_code} for {url}")
                return None
                
            ct = resp.headers.get("content-type", "")
            if "text/html" not in ct.lower():
                return None
                
            return resp
            
        except requests.exceptions.Timeout:
            print(f"Timeout for {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {e}")
        except Exception as e:
            print(f"Unexpected error for {url}: {e}")
        
        return None
    
    def extract_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract and normalize links from page."""
        links = []
        
        for a in soup.find_all("a", href=True):
            link = normalize_url(url, a.get("href"))
            if link and link not in self.seen_urls:
                # Apply domain and path restrictions
                if self.config.restrict_to_domain and not same_domain(link, url):
                    continue
                if self.config.restrict_to_path and not under_path(link, url):
                    continue
                    
                links.append(link)
        
        return links
    
    def crawl_single_url(self, url: str, depth: int = 0) -> Optional[Page]:
        """Crawl a single URL and return Page object if successful."""
        if url in self.seen_urls:
            self.stats.duplicate_urls += 1
            return None
            
        self.seen_urls.add(url)
        self.stats.urls_processed += 1
        
        # Try regular fetch first
        resp = self.fetch_page(url)
        
        # If regular fetch fails and JS is enabled, try JS rendering
        if not resp and self.config.use_javascript:
            resp = self.fetch_page(url, use_js=True)
        
        if not resp:
            self.stats.pages_failed += 1
            return None
        
        # Parse content
        try:
            soup = BeautifulSoup(resp.text, "lxml")
            title, content, meta_desc = extract_main_content(soup, url)
            
            # Filter out pages with insufficient content
            if len(content.strip()) < MIN_CONTENT_LENGTH:
                return None
                
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "..."
            
            # Create page object
            page = Page(
                url=url,
                title=title or "",
                content=content,
                content_type="markdown",
                meta_description=meta_desc or ""
            )
            
            # Check for duplicate content
            if page.content_hash in self.seen_content_hashes:
                self.stats.duplicate_urls += 1
                return None
                
            self.seen_content_hashes.add(page.content_hash)
            self.stats.pages_successful += 1
            
            return page
            
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            self.stats.pages_failed += 1
            return None
    
    def crawl(self, start_url: str) -> List[Page]:
        """Main crawling method with multiple discovery strategies."""
        print(f"Starting robust crawl of {start_url}")
        print(f"Config: max_pages={self.config.max_pages}, use_js={self.config.use_javascript}")
        
        # Normalize start URL
        start_url = start_url.rstrip('/')
        base_domain = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
        
        # Setup robots.txt parser
        rp = robotparser.RobotFileParser()
        robots_content = ""
        if self.config.respect_robots_txt:
            try:
                robots_url = urljoin(base_domain, "/robots.txt")
                rp.set_url(robots_url)
                rp.read()
                
                # Also get content for sitemap discovery
                robots_resp = self.session.get(robots_url, timeout=10)
                if robots_resp.status_code == 200:
                    robots_content = robots_resp.text
                    
                print(f"Loaded robots.txt from {robots_url}")
            except Exception as e:
                print(f"Could not load robots.txt: {e}")
        
        # Discover sitemaps
        queue = [start_url]
        if self.config.enable_sitemap_discovery:
            sitemaps = discover_sitemaps(base_domain, self.session, robots_content)
            if sitemaps:
                print(f"Found {len(sitemaps)} sitemaps")
                
                # Process sitemaps
                for sitemap_url in sitemaps[:3]:  # Limit to first 3
                    print(f"Processing sitemap: {sitemap_url}")
                    sitemap_urls = parse_sitemap(sitemap_url, self.session)
                    
                    # Filter and add sitemap URLs
                    valid_urls = []
                    for url in sitemap_urls:
                        if self.config.restrict_to_domain and not same_domain(url, start_url):
                            continue
                        if self.config.restrict_to_path and not under_path(url, start_url):
                            continue
                        valid_urls.append(url)
                    
                    queue.extend(valid_urls)
                    self.stats.urls_discovered += len(valid_urls)
                    
                    print(f"Added {len(valid_urls)} URLs from {sitemap_url}")
        
        # Remove duplicates while preserving order
        seen_in_queue = set()
        unique_queue = []
        for url in queue:
            if url not in seen_in_queue:
                seen_in_queue.add(url)
                unique_queue.append(url)
        queue = unique_queue
        
        print(f"Starting crawl with {len(queue)} URLs in queue")
        
        pages: List[Page] = []
        
        # Process URLs
        for i, url in enumerate(queue):
            if len(pages) >= self.config.max_pages:
                break
                
            # Check robots.txt
            if self.config.respect_robots_txt and not is_allowed_by_robots(url, rp, self.config):
                continue
            
            # Progress reporting
            if i % 10 == 0 and i > 0:
                print(f"Processed {i} URLs, found {len(pages)} valid pages")
            
            # Random delay for politeness
            if self.config.random_delays and i > 0:
                delay = self.config.sleep_between_requests + random.uniform(0, 1)
                time.sleep(delay)
            else:
                time.sleep(self.config.sleep_between_requests)
            
            # Crawl the URL
            page = self.crawl_single_url(url)
            if page:
                pages.append(page)
                print(f"✓ Scraped: {url} - {page.title[:60]}...")
        
        # Print final statistics
        print(f"\nCrawl completed!")
        print(f"URLs discovered: {self.stats.urls_discovered}")
        print(f"URLs processed: {self.stats.urls_processed}")
        print(f"Pages successful: {self.stats.pages_successful}")
        print(f"Pages failed: {self.stats.pages_failed}")
        print(f"JS rendered pages: {self.stats.js_rendered_pages}")
        print(f"Duplicate URLs skipped: {self.stats.duplicate_urls}")
        
        return pages

# ------------------ Chunking (same as before) ------------------
def md_split(docs, chunk=800, overlap=80):
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"), ("##","h2"), ("###","h3")])
    rc     = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    out: List[Document] = []
    print("Chunking text …")
    t0 = time.time()
    for doc in docs:
        for sec in header.split_text(doc.page_content):
            for ch in rc.split_text(sec.page_content):
                md = dict(doc.metadata)
                md["source"] = md.get("source") or md.get("url")
                out.append(Document(page_content=ch, metadata=md))
    print(f"{len(out)} chunks ({time.time()-t0:.1f}s)\n")
    return out

# ------------------ Build Chroma (same as before) ------------------
def build_stores_from_pages(pages: List[Page], client, chunk_size: int = 800, chunk_overlap: int = 80):
    """Convert pages to documents, chunk, embed, and store."""
    docs: List[Document] = []
    for page in pages:
        meta = {
            "url": page.url,
            "title": page.title,
            "meta_description": page.meta_description,
            "content_type": page.content_type,
            "language": page.language
        }
        # Put the page title as an H1 for header splitter
        content = (f"# {page.title}\n\n" if page.title else "") + page.content
        docs.append(Document(page_content=content, metadata=meta))

    # Chunk with custom sizes
    def md_split_custom(docs):
        header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"), ("##","h2"), ("###","h3")])
        rc = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        out: List[Document] = []
        print("Chunking text …")
        t0 = time.time()
        for doc in docs:
            for sec in header.split_text(doc.page_content):
                for ch in rc.split_text(sec.page_content):
                    md = dict(doc.metadata)
                    md["source"] = md.get("source") or md.get("url")
                    out.append(Document(page_content=ch, metadata=md))
        print(f"{len(out)} chunks ({time.time()-t0:.1f}s)\n")
        return out

    chunks = md_split_custom(docs)

    # Write to Chroma
    print("Embedding & writing to Chroma …")
    txt_col = client.get_or_create_collection("text", embedding_function=SciEmbedding())

    DOC_BATCH = 16
    pbar_text = tqdm(total=len(chunks), desc="Text embeddings", unit="doc")
    next_id_base = txt_col.count()

    for start in range(0, len(chunks), DOC_BATCH):
        batch = chunks[start:start + DOC_BATCH]
        batch_docs = [d.page_content for d in batch]
        batch_metas = [d.metadata for d in batch]
        batch_ids = [f"t{next_id_base + i}" for i in range(len(batch))]

        batch_vecs = embed_texts_token_safe_batch(batch_docs)

        txt_col.add(
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_vecs,
            ids=batch_ids,
        )

        next_id_base += len(batch)
        pbar_text.update(len(batch))

    pbar_text.close()
    print("Vector DB ready\n")

# ------------------ CLI ------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust universal website crawler → Together embeddings → Chroma DB")
    
    # Basic options
    parser.add_argument("--url", required=True, help="Start URL to crawl")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to crawl")
    parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth")
    parser.add_argument("--chroma", type=str, default=str(CHROMA_DIR), help="Chroma persistent dir")
    parser.add_argument("--chunk", type=int, default=800, help="Chunk size for text splitting")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap")
    
    # Crawling behavior
    parser.add_argument("--no-js", action="store_true", help="Disable JavaScript rendering")
    parser.add_argument("--no-sitemap", action="store_true", help="Disable sitemap discovery")  
    parser.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt")
    parser.add_argument("--restrict-domain", action="store_true", help="Stay within same domain")
    parser.add_argument("--restrict-path", action="store_true", help="Stay within same path")
    
    # Performance options
    parser.add_argument("--timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="Request timeout")
    parser.add_argument("--delay", type=float, default=DEFAULT_SLEEP_BETWEEN_REQUESTS, help="Delay between requests")
    parser.add_argument("--no-random-delays", action="store_true", help="Disable random delays")
    parser.add_argument("--no-ua-rotation", action="store_true", help="Disable User-Agent rotation")
    
    args = parser.parse_args()
    
    # Create crawl configuration
    config = CrawlConfig(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        restrict_to_domain=args.restrict_domain,
        restrict_to_path=args.restrict_path,
        use_javascript=not args.no_js,
        respect_robots_txt=not args.ignore_robots,
        enable_sitemap_discovery=not args.no_sitemap,
        request_timeout=args.timeout,
        sleep_between_requests=args.delay,
        user_agent_rotation=not args.no_ua_rotation,
        random_delays=not args.no_random_delays
    )
    
    print(f"🚀 Starting robust crawl of {args.url}")
    print(f"📊 Max pages: {config.max_pages}, JS: {config.use_javascript}, Sitemaps: {config.enable_sitemap_discovery}")
    
    # Crawl the website
    with RobustWebCrawler(config) as crawler:
        pages = crawler.crawl(args.url)
    
    if not pages:
        print("❌ No pages collected; exiting.")
        return
    
    print(f"✅ Collected {len(pages)} pages")
    
    # Store in Chroma
    chroma_dir = Path(args.chroma)
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    build_stores_from_pages(pages, client, chunk_size=args.chunk, chunk_overlap=args.overlap)
    
    print(f"🎉 Successfully crawled and stored {len(pages)} pages in {chroma_dir}")

if __name__ == "__main__":
    main()
