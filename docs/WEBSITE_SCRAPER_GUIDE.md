# �� Perspective D<cide> Website Scraper

A robust CLI tool for scraping entire websites with advanced features including intelligent crawling, content extraction, rate limiting, and multiple output formats.

## 🎯 **What This Tool Does**

The Website Scraper provides comprehensive website crawling capabilities:

- **Intelligent Crawling**: Discovers and follows links automatically
- **Content Extraction**: Clean text extraction with metadata
- **Multiple Engines**: Both aiohttp and Playwright for different site types
- **Rate Limiting**: Polite scraping with configurable delays
- **Robots.txt Compliance**: Respects website crawling policies
- **Error Handling**: Robust error recovery and logging
- **Multiple Formats**: JSONL, JSON, CSV, Parquet output

## 🚀 **Quick Start**

### **Basic Usage**
```bash
# Scrape a website
python website_scraper.py --url https://example.com --output results.jsonl

# Scrape with limits
python website_scraper.py --url https://example.com --max-pages 50 --max-depth 2 --output results.jsonl

# Use Playwright for JavaScript-heavy sites
python website_scraper.py --url https://example.com --use-playwright --output results.jsonl
```

### **Advanced Usage**
```bash
# Scrape with custom settings
python website_scraper.py \
  --url https://example.com \
  --max-pages 100 \
  --max-depth 3 \
  --delay 2.0 \
  --use-playwright \
  --format csv \
  --output website_data.csv

# Scrape with verbose logging
python website_scraper.py \
  --url https://example.com \
  --output results.jsonl \
  --verbose
```

## �� **Supported Output Formats**

### **JSONL (Default)**
```json
{"url": "https://example.com", "content": "Extracted text content...", "metadata": {"title": "Page Title", "word_count": 150}, "extracted_at": "2024-01-15T10:30:00"}
{"url": "https://example.com/about", "content": "About page content...", "metadata": {"title": "About Us", "word_count": 200}, "extracted_at": "2024-01-15T10:30:05"}
```

### **JSON**
```json
[
  {
    "url": "https://example.com",
    "content": "Extracted text content...",
    "metadata": {
      "title": "Page Title",
      "description": "Page description",
      "keywords": "key, words",
      "author": "Author Name",
      "language": "en",
      "word_count": 150
    },
    "raw_html": "<html>...</html>",
    "extracted_at": "2024-01-15T10:30:00"
  }
]
```

### **CSV**
```csv
url,title,description,word_count,language,extracted_at
https://example.com,Page Title,Page description,150,en,2024-01-15T10:30:00
https://example.com/about,About Us,About page description,200,en,2024-01-15T10:30:05
```

### **Parquet**
- Columnar format for efficient storage
- Compression support
- Schema preservation

## �� **Key Features**

### **Intelligent Crawling**
- **Link Discovery**: Automatically finds and follows links
- **Depth Control**: Configurable crawl depth
- **Domain Filtering**: Stays within the target domain
- **URL Normalization**: Handles relative and absolute URLs

### **Content Extraction**
- **Trafilatura Integration**: Advanced content extraction
- **BeautifulSoup Fallback**: Reliable fallback extraction
- **Metadata Extraction**: Title, description, keywords, author
- **Content Cleaning**: Removes scripts, styles, and noise

### **Multiple Engines**
- **aiohttp**: Fast, lightweight for static content
- **Playwright**: Full browser engine for JavaScript-heavy sites
- **Automatic Selection**: Choose based on site requirements

### **Rate Limiting & Politeness**
- **Configurable Delays**: Respectful crawling speeds
- **Robots.txt Compliance**: Follows website policies
- **User-Agent Rotation**: Professional browser identification
- **Error Recovery**: Handles temporary failures gracefully

### **Error Handling**
- **Failed URL Tracking**: Logs and reports failed requests
- **Retry Logic**: Automatic retry for transient failures
- **Graceful Degradation**: Continues despite individual failures
- **Comprehensive Logging**: Detailed error reporting

## �� **Use Cases**

### **1. Content Analysis**
```bash
# Scrape blog for content analysis
python website_scraper.py \
  --url https://blog.example.com \
  --max-pages 200 \
  --output blog_content.jsonl
```

### **2. Research Data Collection**
```bash
# Scrape research papers and articles
python website_scraper.py \
  --url https://research.example.com \
  --max-depth 4 \
  --use-playwright \
  --output research_data.csv
```

### **3. News Monitoring**
```bash
# Scrape news website
python website_scraper.py \
  --url https://news.example.com \
  --max-pages 50 \
  --delay 2.0 \
  --output news_articles.jsonl
```

### **4. E-commerce Product Scraping**
```bash
# Scrape product catalog
python website_scraper.py \
  --url https://store.example.com \
  --max-pages 100 \
  --use-playwright \
  --output products.csv
```

## 🛠️ **Configuration**

### **Configuration File**
```yaml
# configs/website_scraper_config.yaml
core:
  max_pages: 100
  max_depth: 3
  delay_between_requests: 1.0

crawling:
  respect_robots_txt: true
  follow_redirects: true
  user_agent: "Mozilla/5.0..."

extraction:
  use_trafilatura: true
  extract_metadata: true
  clean_html: true

playwright:
  headless: true
  wait_for_network_idle: true
  timeout: 30000
```

### **Environment Variables**
```bash
export PERSPECTIVE_DCIDE_SCRAPER_CONFIG="configs/website_scraper_config.yaml"
export PERSPECTIVE_DCIDE_LOG_LEVEL="INFO"
export PERSPECTIVE_DCIDE_USER_AGENT="Custom User Agent"
```

## 🔍 **Advanced Features**

### **Content Filtering**
```bash
# Filter by content length
python website_scraper.py \
  --url https://example.com \
  --output filtered_results.jsonl \
  --min-content-length 100 \
  --max-content-length 10000
```

### **URL Pattern Matching**
```bash
# Scrape specific URL patterns
python website_scraper.py \
  --url https://example.com \
  --url-pattern "*/blog/*" \
  --output blog_posts.jsonl
```

### **Custom Headers**
```bash
# Add custom headers
python website_scraper.py \
  --url https://example.com \
  --header "Authorization: Bearer token" \
  --header "Accept-Language: en-US" \
  --output results.jsonl
```

### **Session Management**
```bash
# Maintain session across requests
python website_scraper.py \
  --url https://example.com \
  --maintain-session \
  --cookie "session=abc123" \
  --output results.jsonl
```

## 📈 **Performance Optimization**

### **Concurrent Scraping**
```bash
# Increase concurrency for faster scraping
python website_scraper.py \
  --url https://example.com \
  --max-concurrent 10 \
  --output results.jsonl
```

### **Batch Processing**
```bash
# Process multiple sites
for site in sites.txt; do
  python website_scraper.py \
    --url "$site" \
    --output "results/${site//\//_}.jsonl"
done
```

### **Incremental Scraping**
```bash
# Resume from previous state
python website_scraper.py \
  --url https://example.com \
  --resume-from state.json \
  --output results.jsonl
```

## �� **Ethical Scraping**

### **Best Practices**
- **Respect robots.txt**: Always check and follow robots.txt
- **Rate Limiting**: Use appropriate delays between requests
- **User-Agent**: Use descriptive user-agent strings
- **Terms of Service**: Check website terms before scraping
- **Data Usage**: Only collect data you have permission to use

### **Legal Considerations**
- **Copyright**: Respect copyright and intellectual property
- **Privacy**: Don't collect personal information without consent
- **Terms of Service**: Follow website terms and conditions
- **Rate Limits**: Don't overwhelm servers with requests

## 🚨 **Error Handling**

### **Common Issues**
- **Connection Timeouts**: Network connectivity problems
- **Rate Limiting**: Server-side rate limiting
- **CAPTCHA**: Anti-bot protection
- **JavaScript**: Dynamic content loading
- **Authentication**: Login-required content

### **Solutions**
- **Retry Logic**: Automatic retry with exponential backoff
- **Proxy Rotation**: Use multiple IP addresses
- **User-Agent Rotation**: Vary browser identification
- **Session Management**: Maintain login sessions
- **CAPTCHA Handling**: Manual or automated CAPTCHA solving

## 📊 **Output Analysis**

### **Content Statistics**
```python
import pandas as pd

# Load scraped data
df = pd.read_json('results.jsonl', lines=True)

# Basic statistics
print(f"Total pages: {len(df)}")
print(f"Total words: {df['metadata'].apply(lambda x: x['word_count']).sum()}")
print(f"Average words per page: {df['metadata'].apply(lambda x: x['word_count']).mean()}")

# Language distribution
languages = df['metadata'].apply(lambda x: x.get('language', 'unknown'))
print(languages.value_counts())
```

### **Content Analysis**
```python
# Extract key insights
titles = df['metadata'].apply(lambda x: x.get('title', ''))
descriptions = df['metadata'].apply(lambda x: x.get('description', ''))

# Find most common words
from collections import Counter
import re

all_text = ' '.join(df['content'].tolist())
words = re.findall(r'\b\w+\b', all_text.lower())
word_counts = Counter(words)
print("Most common words:", word_counts.most_common(10))
```

## 🔮 **Future Enhancements**

- **Sitemap Integration**: Parse XML sitemaps for better discovery
- **RSS Feed Support**: Extract content from RSS feeds
- **API Discovery**: Automatically find and use APIs
- **Form Handling**: Fill out and submit forms
- **Login Support**: Handle authentication and sessions
- **CAPTCHA Solving**: Automated CAPTCHA resolution
- **Image Extraction**: Download and process images
- **Video Processing**: Extract video content and metadata

## 📚 **Documentation**

- **`website_scraper.py`**: Main CLI tool
- **`website_scraper_demo.py`**: Comprehensive demonstration
- **`website_scraper_config.yaml`**: Configuration options
- **`WEBSITE_SCRAPER_GUIDE.md`**: This guide

## �� **Contributing**

The Website Scraper is designed to be extensible. You can:
- Add new content extraction methods
- Implement custom URL filters
- Create new output formats
- Enhance error handling
- Add authentication methods

---

**🌐 Ready to scrape websites robustly and ethically!** 