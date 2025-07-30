#!/usr/bin/env python3
"""
Perspective D<cide> Website Scraper

A robust CLI tool for scraping entire websites with advanced features including:
- Intelligent crawling and discovery
- Content extraction and cleaning
- Rate limiting and politeness
- Multiple output formats
- Error handling and recovery
- Progress tracking and logging
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import pandas as pd
import polars as pl
from playwright.async_api import async_playwright
import trafilatura
from urllib.robotparser import RobotFileParser
import hashlib
import re

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from perspective_dcide.core.schemas import ContentItem
from perspective_dcide.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class WebsiteScraper:
    """Robust website scraper with advanced features."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.session = None
        self.browser = None
        self.page = None
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.robots_parser = None
        self.rate_limiter = RateLimiter()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize scraper components."""
        # Initialize aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        # Initialize Playwright for JavaScript-heavy sites
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        self.page = await self.browser.new_page()
        
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def setup_robots_parser(self, base_url: str):
        """Setup robots.txt parser."""
        try:
            robots_url = urljoin(base_url, '/robots.txt')
            self.robots_parser = RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
        except Exception as e:
            logger.warning(f"Could not read robots.txt: {e}")
            self.robots_parser = None
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.robots_parser:
            return True
        return self.robots_parser.can_fetch('*', url)
    
    def normalize_url(self, url: str, base_url: str) -> str:
        """Normalize URL and remove fragments."""
        parsed = urlparse(url)
        if not parsed.scheme:
            url = urljoin(base_url, url)
        # Remove fragments and query parameters if needed
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ['http', 'https'] and
                parsed.netloc == base_domain and
                not any(ext in parsed.path.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js'])
            )
        except:
            return False
    
    async def get_page_content(self, url: str, use_playwright: bool = False) -> Optional[Dict[str, Any]]:
        """Get page content using either aiohttp or Playwright."""
        
        await self.rate_limiter.wait()
        
        try:
            if use_playwright:
                # Use Playwright for JavaScript-heavy sites
                await self.page.goto(url, wait_until='networkidle', timeout=30000)
                content = await self.page.content()
                title = await self.page.title()
                return {
                    'url': url,
                    'content': content,
                    'title': title,
                    'status': 200
                }
            else:
                # Use aiohttp for static content
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            'url': url,
                            'content': content,
                            'title': '',  # Will be extracted from HTML
                            'status': response.status
                        }
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            self.failed_urls.add(url)
            return None
    
    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links.append(full_url)
        
        return links
    
    def extract_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract clean content from HTML."""
        
        # Use trafilatura for main content extraction
        extracted_text = trafilatura.extract(html_content, include_links=True, include_images=True)
        
        # Fallback to BeautifulSoup if trafilatura fails
        if not extracted_text:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else ''
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                extracted_text = main_content.get_text(separator='\n', strip=True)
            else:
                extracted_text = soup.get_text(separator='\n', strip=True)
        
        # Extract metadata
        soup = BeautifulSoup(html_content, 'html.parser')
        metadata = {
            'title': soup.find('title').get_text() if soup.find('title') else '',
            'description': '',
            'keywords': '',
            'author': '',
            'language': soup.get('lang', ''),
            'word_count': len(extracted_text.split()) if extracted_text else 0
        }
        
        # Extract meta tags
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author:
            metadata['author'] = meta_author.get('content', '')
        
        return {
            'url': url,
            'content': extracted_text or '',
            'metadata': metadata,
            'raw_html': html_content,
            'extracted_at': datetime.utcnow().isoformat()
        }
    
    async def crawl_website(self, start_url: str, max_pages: int = 100, max_depth: int = 3, 
                           use_playwright: bool = False, delay: float = 1.0) -> List[Dict[str, Any]]:
        """Crawl entire website and extract content."""
        
        console.print(f"🌐 Starting website crawl: {start_url}")
        console.print(f"📊 Max pages: {max_pages}, Max depth: {max_depth}")
        
        # Setup
        base_domain = urlparse(start_url).netloc
        self.setup_robots_parser(start_url)
        self.rate_limiter.delay = delay
        
        # Initialize crawl queue
        to_visit = [(start_url, 0)]  # (url, depth)
        scraped_pages = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Crawling website...", total=max_pages)
            
            while to_visit and len(scraped_pages) < max_pages:
                current_url, depth = to_visit.pop(0)
                
                # Skip if already visited or too deep
                if current_url in self.visited_urls or depth > max_depth:
                    continue
                
                # Check robots.txt
                if not self.can_fetch(current_url):
                    logger.info(f"Skipping {current_url} (robots.txt)")
                    continue
                
                self.visited_urls.add(current_url)
                
                # Get page content
                page_data = await self.get_page_content(current_url, use_playwright)
                if not page_data:
                    continue
                
                # Extract content
                extracted_data = self.extract_content(page_data['content'], current_url)
                scraped_pages.append(extracted_data)
                
                progress.update(task, completed=len(scraped_pages))
                progress.update(task, description=f"Crawling... ({len(scraped_pages)}/{max_pages})")
                
                # Extract links for next level
                if depth < max_depth:
                    links = self.extract_links(page_data['content'], current_url)
                    for link in links:
                        normalized_link = self.normalize_url(link, start_url)
                        if (self.is_valid_url(normalized_link, base_domain) and 
                            normalized_link not in self.visited_urls):
                            to_visit.append((normalized_link, depth + 1))
        
        console.print(f"✅ Crawl completed! Scraped {len(scraped_pages)} pages")
        console.print(f"❌ Failed URLs: {len(self.failed_urls)}")
        
        return scraped_pages
    
    def save_results(self, scraped_pages: List[Dict[str, Any]], output_path: str, format: str = "jsonl"):
        """Save scraped results to file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "jsonl":
                with open(output_file, 'w', encoding='utf-8') as f:
                    for page in scraped_pages:
                        f.write(json.dumps(page, ensure_ascii=False) + '\n')
                        
            elif format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(scraped_pages, f, ensure_ascii=False, indent=2)
                    
            elif format.lower() == "csv":
                # Create a simplified CSV with key fields
                df_data = []
                for page in scraped_pages:
                    df_data.append({
                        'url': page['url'],
                        'title': page['metadata']['title'],
                        'description': page['metadata']['description'],
                        'word_count': page['metadata']['word_count'],
                        'language': page['metadata']['language'],
                        'extracted_at': page['extracted_at']
                    })
                df = pd.DataFrame(df_data)
                df.to_csv(output_file, index=False, encoding='utf-8')
                
            elif format.lower() == "parquet":
                # Create a simplified DataFrame
                df_data = []
                for page in scraped_pages:
                    df_data.append({
                        'url': page['url'],
                        'title': page['metadata']['title'],
                        'description': page['metadata']['description'],
                        'word_count': page['metadata']['word_count'],
                        'language': page['metadata']['language'],
                        'extracted_at': page['extracted_at']
                    })
                df = pl.DataFrame(df_data)
                df.write_parquet(output_file)
            
            console.print(f"✅ Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def display_summary(self, scraped_pages: List[Dict[str, Any]]):
        """Display scraping summary."""
        
        if not scraped_pages:
            console.print("❌ No pages were successfully scraped")
            return
        
        # Create summary table
        summary_table = Table(title="Scraping Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        total_pages = len(scraped_pages)
        total_words = sum(page['metadata']['word_count'] for page in scraped_pages)
        avg_words = total_words / total_pages if total_pages > 0 else 0
        
        # Language distribution
        languages = {}
        for page in scraped_pages:
            lang = page['metadata']['language'] or 'unknown'
            languages[lang] = languages.get(lang, 0) + 1
        
        summary_table.add_row("Total Pages", str(total_pages))
        summary_table.add_row("Total Words", f"{total_words:,}")
        summary_table.add_row("Average Words/Page", f"{avg_words:.1f}")
        summary_table.add_row("Failed URLs", str(len(self.failed_urls)))
        summary_table.add_row("Languages", str(len(languages)))
        
        console.print(summary_table)
        
        # Show sample pages
        sample_table = Table(title="Sample Pages")
        sample_table.add_column("URL", style="cyan")
        sample_table.add_column("Title", style="green")
        sample_table.add_column("Words", style="yellow")
        sample_table.add_column("Language", style="magenta")
        
        for page in scraped_pages[:5]:  # Show first 5
            sample_table.add_row(
                page['url'][:50] + "..." if len(page['url']) > 50 else page['url'],
                page['metadata']['title'][:30] + "..." if len(page['metadata']['title']) > 30 else page['metadata']['title'],
                str(page['metadata']['word_count']),
                page['metadata']['language'] or 'unknown'
            )
        
        console.print(sample_table)

class RateLimiter:
    """Simple rate limiter for polite scraping."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_request = 0
    
    async def wait(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        time_since_last = now - self.last_request
        if time_since_last < self.delay:
            await asyncio.sleep(self.delay - time_since_last)
        self.last_request = time.time()

async def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Perspective D<cide> Website Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python website_scraper.py --url https://example.com --output results.jsonl
  python website_scraper.py --url https://example.com --max-pages 50 --format csv
  python website_scraper.py --url https://example.com --use-playwright --delay 2.0
        """
    )
    
    parser.add_argument("--url", "-u", required=True, help="Starting URL to scrape")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--format", "-f", default="jsonl", choices=["jsonl", "json", "csv", "parquet"], help="Output format")
    parser.add_argument("--max-pages", "-m", type=int, default=100, help="Maximum pages to scrape")
    parser.add_argument("--max-depth", "-d", type=int, default=3, help="Maximum crawl depth")
    parser.add_argument("--use-playwright", "-p", action="store_true", help="Use Playwright for JavaScript-heavy sites")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Display header
        console.print(Panel.fit(
            "[bold blue]Perspective D<cide> Website Scraper[/bold blue]\n"
            "[dim]Robust website scraping with intelligent crawling[/dim]",
            border_style="blue"
        ))
        
        # Initialize and run scraper
        async with WebsiteScraper() as scraper:
            # Crawl website
            scraped_pages = await scraper.crawl_website(
                start_url=args.url,
                max_pages=args.max_pages,
                max_depth=args.max_depth,
                use_playwright=args.use_playwright,
                delay=args.delay
            )
            
            # Display summary
            scraper.display_summary(scraped_pages)
            
            # Save results
            scraper.save_results(scraped_pages, args.output, args.format)
        
        console.print(f"\n✅ Website scraping completed successfully!")
        
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 