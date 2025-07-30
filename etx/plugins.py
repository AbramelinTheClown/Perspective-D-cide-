"""
ETX Plugins for Perspective D<cide>.

Provides plugin functionality for extending the Emergent Taxonomy framework.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)

@dataclass
class PluginMetadata:
    """Metadata for ETX plugins."""
    
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}

class BaseETXPlugin(ABC):
    """Base class for ETX plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metadata = self.get_metadata()
        self._validate_config()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    def process_content(self, content: List[str], embeddings: Any = None) -> Dict[str, Any]:
        """Process content and return results."""
        pass
    
    def _validate_config(self) -> None:
        """Validate plugin configuration."""
        schema = self.metadata.config_schema
        if not schema:
            return
        
        for key, required in schema.items():
            if required and key not in self.config:
                raise ValueError(f"Required config key '{key}' not found for plugin {self.metadata.name}")
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.config.get("enabled", True)

class KeywordExtractionPlugin(BaseETXPlugin):
    """Plugin for extracting keywords from content."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="keyword_extraction",
            version="1.0.0",
            description="Extracts keywords from content using TF-IDF",
            author="Perspective D<cide> Team",
            dependencies=["scikit-learn"],
            config_schema={
                "max_keywords": False,
                "min_tf": False,
                "stop_words": False
            }
        )
    
    def process_content(self, content: List[str], embeddings: Any = None) -> Dict[str, Any]:
        """Extract keywords from content."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            
            max_keywords = self.config.get("max_keywords", 10)
            min_tf = self.config.get("min_tf", 0.1)
            stop_words = self.config.get("stop_words", list(ENGLISH_STOP_WORDS))
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_keywords * 2,
                min_df=min_tf,
                stop_words=stop_words,
                ngram_range=(1, 2)
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(content)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = tfidf_scores.argsort()[-max_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            return {
                "keywords": keywords,
                "scores": tfidf_scores[top_indices].tolist(),
                "method": "tfidf"
            }
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple keyword extraction")
            return self._simple_keyword_extraction(content)
    
    def _simple_keyword_extraction(self, content: List[str]) -> Dict[str, Any]:
        """Simple keyword extraction fallback."""
        import re
        
        # Combine all content
        all_text = " ".join(content).lower()
        
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', all_text)
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        max_keywords = self.config.get("max_keywords", 10)
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        
        return {
            "keywords": [word for word, count in keywords],
            "scores": [count for word, count in keywords],
            "method": "frequency"
        }

class SentimentAnalysisPlugin(BaseETXPlugin):
    """Plugin for sentiment analysis."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="sentiment_analysis",
            version="1.0.0",
            description="Analyzes sentiment of content",
            author="Perspective D<cide> Team",
            dependencies=["textblob"],
            config_schema={
                "language": False,
                "threshold": False
            }
        )
    
    def process_content(self, content: List[str], embeddings: Any = None) -> Dict[str, Any]:
        """Analyze sentiment of content."""
        try:
            from textblob import TextBlob
            
            sentiments = []
            polarities = []
            subjectivities = []
            
            for text in content:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment)
                polarities.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            
            # Calculate overall sentiment
            avg_polarity = sum(polarities) / len(polarities) if polarities else 0
            avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
            
            # Classify overall sentiment
            threshold = self.config.get("threshold", 0.1)
            if avg_polarity > threshold:
                overall_sentiment = "positive"
            elif avg_polarity < -threshold:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "avg_polarity": avg_polarity,
                "avg_subjectivity": avg_subjectivity,
                "sentiments": [
                    {
                        "polarity": pol,
                        "subjectivity": sub
                    }
                    for pol, sub in zip(polarities, subjectivities)
                ],
                "method": "textblob"
            }
            
        except ImportError:
            logger.warning("textblob not available, using simple sentiment analysis")
            return self._simple_sentiment_analysis(content)
    
    def _simple_sentiment_analysis(self, content: List[str]) -> Dict[str, Any]:
        """Simple sentiment analysis fallback."""
        # Simple positive/negative word counting
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "happy", "joy", "success", "win", "best"
        }
        negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "sad", "angry", "fail", "lose", "worst", "problem", "error"
        }
        
        sentiments = []
        total_positive = 0
        total_negative = 0
        
        for text in content:
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_positive += positive_count
            total_negative += negative_count
            
            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            sentiments.append({
                "polarity": (positive_count - negative_count) / max(len(words), 1),
                "subjectivity": 0.5,  # Default subjectivity
                "sentiment": sentiment
            })
        
        # Calculate overall sentiment
        if total_positive > total_negative:
            overall_sentiment = "positive"
        elif total_negative > total_positive:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "avg_polarity": (total_positive - total_negative) / max(total_positive + total_negative, 1),
            "avg_subjectivity": 0.5,
            "sentiments": sentiments,
            "method": "word_counting"
        }

class LanguageDetectionPlugin(BaseETXPlugin):
    """Plugin for language detection."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="language_detection",
            version="1.0.0",
            description="Detects language of content",
            author="Perspective D<cide> Team",
            dependencies=["langdetect"],
            config_schema={
                "confidence_threshold": False
            }
        )
    
    def process_content(self, content: List[str], embeddings: Any = None) -> Dict[str, Any]:
        """Detect language of content."""
        try:
            from langdetect import detect, detect_langs, LangDetectException
            
            languages = []
            confidence_threshold = self.config.get("confidence_threshold", 0.5)
            
            for text in content:
                try:
                    # Get primary language
                    primary_lang = detect(text)
                    
                    # Get language probabilities
                    lang_probs = detect_langs(text)
                    primary_prob = next((prob.prob for prob in lang_probs if prob.lang == primary_lang), 0.0)
                    
                    if primary_prob >= confidence_threshold:
                        languages.append({
                            "language": primary_lang,
                            "confidence": primary_prob,
                            "all_languages": [
                                {"lang": prob.lang, "prob": prob.prob}
                                for prob in lang_probs
                            ]
                        })
                    else:
                        languages.append({
                            "language": "unknown",
                            "confidence": 0.0,
                            "all_languages": []
                        })
                        
                except LangDetectException:
                    languages.append({
                        "language": "unknown",
                        "confidence": 0.0,
                        "all_languages": []
                    })
            
            # Get most common language
            lang_counts = {}
            for lang_info in languages:
                lang = lang_info["language"]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            most_common_lang = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else "unknown"
            
            return {
                "primary_language": most_common_lang,
                "languages": languages,
                "language_distribution": lang_counts,
                "method": "langdetect"
            }
            
        except ImportError:
            logger.warning("langdetect not available, using simple language detection")
            return self._simple_language_detection(content)
    
    def _simple_language_detection(self, content: List[str]) -> Dict[str, Any]:
        """Simple language detection fallback."""
        # Very basic language detection based on character sets
        languages = []
        
        for text in content:
            # Simple heuristics
            if any(ord(char) > 127 for char in text):
                # Contains non-ASCII characters
                if any(ord(char) > 1000 for char in text):
                    language = "unknown"  # Could be CJK, Arabic, etc.
                else:
                    language = "latin_extended"
            else:
                language = "english"
            
            languages.append({
                "language": language,
                "confidence": 0.5,
                "all_languages": [{"lang": language, "prob": 0.5}]
            })
        
        # Get most common language
        lang_counts = {}
        for lang_info in languages:
            lang = lang_info["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        most_common_lang = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else "unknown"
        
        return {
            "primary_language": most_common_lang,
            "languages": languages,
            "language_distribution": lang_counts,
            "method": "character_heuristics"
        }

class PluginManager:
    """Manages ETX plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, BaseETXPlugin] = {}
        self._register_default_plugins()
    
    def _register_default_plugins(self) -> None:
        """Register default plugins."""
        default_plugins = [
            KeywordExtractionPlugin(),
            SentimentAnalysisPlugin(),
            LanguageDetectionPlugin()
        ]
        
        for plugin in default_plugins:
            self.register_plugin(plugin)
    
    def register_plugin(self, plugin: BaseETXPlugin) -> None:
        """Register a plugin."""
        if not plugin.is_enabled():
            return
        
        self.plugins[plugin.metadata.name] = plugin
        logger.info(f"Registered plugin: {plugin.metadata.name} v{plugin.metadata.version}")
    
    def get_plugin(self, name: str) -> Optional[BaseETXPlugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self.plugins.keys())
    
    def process_with_plugins(self, content: List[str], plugin_names: List[str] = None) -> Dict[str, Any]:
        """Process content with specified plugins."""
        if plugin_names is None:
            plugin_names = list(self.plugins.keys())
        
        results = {}
        
        for plugin_name in plugin_names:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                try:
                    results[plugin_name] = plugin.process_content(content)
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} failed: {e}")
                    results[plugin_name] = {"error": str(e)}
        
        return results
    
    def get_plugin_metadata(self) -> Dict[str, PluginMetadata]:
        """Get metadata for all plugins."""
        return {
            name: plugin.metadata
            for name, plugin in self.plugins.items()
        } 