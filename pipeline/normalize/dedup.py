"""
Deduplication utilities using SimHash, MinHash, and vector similarity.
"""

import hashlib
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from schemas.base import ChunkMetadata
from cli.utils.logging import gola_logger

@dataclass
class DedupConfig:
    """Deduplication configuration."""
    simhash_threshold: int = 3  # Hamming distance threshold
    minhash_threshold: float = 0.8  # Jaccard similarity threshold
    vector_threshold: float = 0.92  # Cosine similarity threshold
    minhash_num_permutations: int = 128
    tfidf_max_features: int = 1000
    enable_simhash: bool = True
    enable_minhash: bool = True
    enable_vector: bool = True
    ngram_size: int = 3

class SimHashDeduplicator:
    """SimHash-based deduplication."""
    
    def __init__(self, config: DedupConfig):
        """
        Initialize SimHash deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config
        self.simhash_index: Dict[int, List[str]] = defaultdict(list)
        
        if not SIMHASH_AVAILABLE:
            gola_logger.warning("SimHash library not available. SimHash deduplication disabled.")
    
    def compute_simhash(self, text: str) -> Optional[int]:
        """
        Compute SimHash for text.
        
        Args:
            text: Text to hash
            
        Returns:
            SimHash value or None if not available
        """
        if not SIMHASH_AVAILABLE:
            return None
        
        try:
            # Tokenize text into n-grams
            tokens = self._tokenize_text(text)
            
            # Compute SimHash
            hash_value = simhash.Simhash(tokens)
            return hash_value.value
        
        except Exception as e:
            gola_logger.error(f"Error computing SimHash: {e}")
            return None
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into n-grams.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of n-grams
        """
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Generate n-grams
        ngrams = []
        for i in range(len(words) - self.config.ngram_size + 1):
            ngram = ' '.join(words[i:i + self.config.ngram_size])
            ngrams.append(ngram)
        
        return ngrams
    
    def find_similar_chunks(self, chunk_hash: str, simhash_value: int) -> List[str]:
        """
        Find chunks with similar SimHash.
        
        Args:
            chunk_hash: Current chunk hash
            simhash_value: SimHash value
            
        Returns:
            List of similar chunk hashes
        """
        if not SIMHASH_AVAILABLE or simhash_value is None:
            return []
        
        similar_chunks = []
        
        # Check against existing SimHashes
        for existing_simhash, chunk_hashes in self.simhash_index.items():
            # Calculate Hamming distance
            distance = bin(simhash_value ^ existing_simhash).count('1')
            
            if distance <= self.config.simhash_threshold:
                similar_chunks.extend(chunk_hashes)
        
        return list(set(similar_chunks))
    
    def add_chunk(self, chunk_hash: str, simhash_value: int) -> None:
        """
        Add chunk to SimHash index.
        
        Args:
            chunk_hash: Chunk hash
            simhash_value: SimHash value
        """
        if SIMHASH_AVAILABLE and simhash_value is not None:
            self.simhash_index[simhash_value].append(chunk_hash)

class MinHashDeduplicator:
    """MinHash-based deduplication."""
    
    def __init__(self, config: DedupConfig):
        """
        Initialize MinHash deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config
        self.minhash_index: Dict[int, List[str]] = defaultdict(list)
        
        if not SKLEARN_AVAILABLE:
            gola_logger.warning("Scikit-learn not available. MinHash deduplication disabled.")
    
    def compute_minhash(self, text: str) -> Optional[List[int]]:
        """
        Compute MinHash signature for text.
        
        Args:
            text: Text to hash
            
        Returns:
            MinHash signature or None if not available
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            # Tokenize text
            tokens = self._tokenize_text(text)
            
            # Create feature vector
            features = set(tokens)
            
            # Simple MinHash implementation
            signature = []
            for i in range(self.config.minhash_num_permutations):
                # Use different hash functions
                hash_values = [hash(f"{token}_{i}") % (2**32) for token in features]
                signature.append(min(hash_values))
            
            return signature
        
        except Exception as e:
            gola_logger.error(f"Error computing MinHash: {e}")
            return None
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into shingles.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of shingles
        """
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Generate shingles
        shingles = []
        for i in range(len(words) - self.config.ngram_size + 1):
            shingle = ' '.join(words[i:i + self.config.ngram_size])
            shingles.append(shingle)
        
        return shingles
    
    def compute_jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Compute Jaccard similarity between MinHash signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Jaccard similarity
        """
        if not sig1 or not sig2:
            return 0.0
        
        # Count matching positions
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def find_similar_chunks(self, chunk_hash: str, minhash_sig: List[int]) -> List[str]:
        """
        Find chunks with similar MinHash signatures.
        
        Args:
            chunk_hash: Current chunk hash
            minhash_sig: MinHash signature
            
        Returns:
            List of similar chunk hashes
        """
        if not SKLEARN_AVAILABLE or not minhash_sig:
            return []
        
        similar_chunks = []
        
        # Check against existing signatures
        for existing_sig, chunk_hashes in self.minhash_index.items():
            similarity = self.compute_jaccard_similarity(minhash_sig, existing_sig)
            
            if similarity >= self.config.minhash_threshold:
                similar_chunks.extend(chunk_hashes)
        
        return list(set(similar_chunks))
    
    def add_chunk(self, chunk_hash: str, minhash_sig: List[int]) -> None:
        """
        Add chunk to MinHash index.
        
        Args:
            chunk_hash: Chunk hash
            minhash_sig: MinHash signature
        """
        if SKLEARN_AVAILABLE and minhash_sig:
            # Use first few values as key for indexing
            key = hash(tuple(minhash_sig[:10]))
            self.minhash_index[key].append(chunk_hash)

class VectorDeduplicator:
    """Vector similarity-based deduplication."""
    
    def __init__(self, config: DedupConfig):
        """
        Initialize vector deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config
        self.vectorizer = None
        self.vectors = []
        self.chunk_hashes = []
        
        if not SKLEARN_AVAILABLE:
            gola_logger.warning("Scikit-learn not available. Vector deduplication disabled.")
        else:
            self._init_vectorizer()
    
    def _init_vectorizer(self) -> None:
        """Initialize TF-IDF vectorizer."""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
        except Exception as e:
            gola_logger.error(f"Error initializing vectorizer: {e}")
            self.vectorizer = None
    
    def compute_vector(self, text: str) -> Optional[np.ndarray]:
        """
        Compute TF-IDF vector for text.
        
        Args:
            text: Text to vectorize
            
        Returns:
            TF-IDF vector or None if not available
        """
        if not SKLEARN_AVAILABLE or not self.vectorizer:
            return None
        
        try:
            # Compute vector
            vector = self.vectorizer.fit_transform([text])
            return vector.toarray()[0]
        
        except Exception as e:
            gola_logger.error(f"Error computing vector: {e}")
            return None
    
    def find_similar_chunks(self, chunk_hash: str, vector: np.ndarray) -> List[str]:
        """
        Find chunks with similar vectors.
        
        Args:
            chunk_hash: Current chunk hash
            vector: TF-IDF vector
            
        Returns:
            List of similar chunk hashes
        """
        if not SKLEARN_AVAILABLE or not self.vectors or vector is None:
            return []
        
        try:
            # Compute similarities
            similarities = cosine_similarity([vector], self.vectors)[0]
            
            # Find similar chunks
            similar_chunks = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.config.vector_threshold:
                    similar_chunks.append(self.chunk_hashes[i])
            
            return similar_chunks
        
        except Exception as e:
            gola_logger.error(f"Error finding similar chunks: {e}")
            return []
    
    def add_chunk(self, chunk_hash: str, vector: np.ndarray) -> None:
        """
        Add chunk to vector index.
        
        Args:
            chunk_hash: Chunk hash
            vector: TF-IDF vector
        """
        if SKLEARN_AVAILABLE and vector is not None:
            self.vectors.append(vector)
            self.chunk_hashes.append(chunk_hash)

class Deduplicator:
    """Main deduplicator combining multiple methods."""
    
    def __init__(self, config: Optional[DedupConfig] = None):
        """
        Initialize deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config or DedupConfig()
        
        # Initialize deduplicators
        self.simhash_dedup = SimHashDeduplicator(self.config) if self.config.enable_simhash else None
        self.minhash_dedup = MinHashDeduplicator(self.config) if self.config.enable_minhash else None
        self.vector_dedup = VectorDeduplicator(self.config) if self.config.enable_vector else None
        
        # Track processed chunks
        self.processed_chunks: Set[str] = set()
        self.duplicate_groups: Dict[str, List[str]] = {}
    
    def deduplicate_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Deduplicate chunks using multiple methods.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            Deduplicated chunks
        """
        if not chunks:
            return []
        
        gola_logger.info(f"Deduplicating {len(chunks)} chunks")
        
        deduplicated_chunks = []
        duplicates_found = 0
        
        for chunk in chunks:
            # Skip if already processed
            if chunk.chunk_hash in self.processed_chunks:
                continue
            
            # Check for duplicates
            duplicate_chunks = self._find_duplicates(chunk)
            
            if duplicate_chunks:
                # Mark as duplicate
                chunk.duplicate_of = duplicate_chunks[0]
                duplicates_found += 1
                
                # Group duplicates
                if duplicate_chunks[0] not in self.duplicate_groups:
                    self.duplicate_groups[duplicate_chunks[0]] = []
                self.duplicate_groups[duplicate_chunks[0]].append(chunk.chunk_hash)
                
                gola_logger.debug(f"Found duplicate: {chunk.chunk_hash} -> {duplicate_chunks[0]}")
            else:
                # Add to deduplicators
                self._add_chunk_to_deduplicators(chunk)
                deduplicated_chunks.append(chunk)
            
            # Mark as processed
            self.processed_chunks.add(chunk.chunk_hash)
        
        gola_logger.info(f"Deduplication complete: {len(deduplicated_chunks)} unique, {duplicates_found} duplicates")
        return deduplicated_chunks
    
    def _find_duplicates(self, chunk: ChunkMetadata) -> List[str]:
        """
        Find duplicates for a chunk using all methods.
        
        Args:
            chunk: Chunk to check
            
        Returns:
            List of duplicate chunk hashes
        """
        duplicates = set()
        
        # SimHash deduplication
        if self.simhash_dedup:
            simhash_value = self.simhash_dedup.compute_simhash(chunk.text_norm)
            if simhash_value:
                chunk.simhash64 = str(simhash_value)
                simhash_duplicates = self.simhash_dedup.find_similar_chunks(chunk.chunk_hash, simhash_value)
                duplicates.update(simhash_duplicates)
        
        # MinHash deduplication
        if self.minhash_dedup:
            minhash_sig = self.minhash_dedup.compute_minhash(chunk.text_norm)
            if minhash_sig:
                chunk.minhash_sig = bytes(minhash_sig)
                minhash_duplicates = self.minhash_dedup.find_similar_chunks(chunk.chunk_hash, minhash_sig)
                duplicates.update(minhash_duplicates)
        
        # Vector deduplication
        if self.vector_dedup:
            vector = self.vector_dedup.compute_vector(chunk.text_norm)
            if vector is not None:
                vector_duplicates = self.vector_dedup.find_similar_chunks(chunk.chunk_hash, vector)
                duplicates.update(vector_duplicates)
        
        return list(duplicates)
    
    def _add_chunk_to_deduplicators(self, chunk: ChunkMetadata) -> None:
        """
        Add chunk to all deduplicators.
        
        Args:
            chunk: Chunk to add
        """
        # Add to SimHash deduplicator
        if self.simhash_dedup and chunk.simhash64:
            simhash_value = int(chunk.simhash64)
            self.simhash_dedup.add_chunk(chunk.chunk_hash, simhash_value)
        
        # Add to MinHash deduplicator
        if self.minhash_dedup and chunk.minhash_sig:
            minhash_sig = list(chunk.minhash_sig)
            self.minhash_dedup.add_chunk(chunk.chunk_hash, minhash_sig)
        
        # Add to vector deduplicator
        if self.vector_dedup:
            vector = self.vector_dedup.compute_vector(chunk.text_norm)
            if vector is not None:
                self.vector_dedup.add_chunk(chunk.chunk_hash, vector)
    
    def get_duplicate_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.
        
        Args:
            Statistics dictionary
        """
        return {
            "total_chunks": len(self.processed_chunks),
            "unique_chunks": len(self.processed_chunks) - len(self.duplicate_groups),
            "duplicate_groups": len(self.duplicate_groups),
            "total_duplicates": sum(len(group) for group in self.duplicate_groups.values()),
            "duplicate_ratio": len(self.duplicate_groups) / len(self.processed_chunks) if self.processed_chunks else 0.0
        }
    
    def get_duplicate_groups(self) -> Dict[str, List[str]]:
        """
        Get duplicate groups.
        
        Returns:
            Dictionary mapping representative chunk to duplicate chunks
        """
        return self.duplicate_groups.copy()

# Global deduplicator instance
deduplicator = Deduplicator()

def get_deduplicator() -> Deduplicator:
    """Get the global deduplicator instance."""
    return deduplicator

def init_deduplicator(config: Optional[DedupConfig] = None) -> Deduplicator:
    """
    Initialize the global deduplicator.
    
    Args:
        config: Deduplication configuration
        
    Returns:
        Deduplicator instance
    """
    global deduplicator
    deduplicator = Deduplicator(config)
    return deduplicator 