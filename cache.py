import os
import psycopg2
import numpy as np
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class SemanticCacheManager:
    """Manages a cache of query-response pairs using PostgreSQL and embedding similarity with variant support and frequency tracking."""

    def __init__(self, similarity_threshold: float = None):
        self.pg_host = os.environ.get('PG_HOST', 'localhost')
        self.pg_port = os.environ.get('PG_PORT', '5432')
        self.pg_dbname = os.environ.get('PG_DBNAME', 'semantic_cache')
        self.pg_user = os.environ.get('PG_USER', 'postgres')
        self.pg_password = os.environ.get('PG_PASSWORD', '')
        self.similarity_threshold = similarity_threshold or float(os.environ.get('SIMILARITY_THRESHOLD', '0.85'))

        self.conn = self.initialize_database()

    def initialize_database(self):
        """Connect to the database and initialize required tables."""
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.pg_dbname,
            user=self.pg_user,
            password=self.pg_password
        )
        cursor = conn.cursor()

        # Create tables with new schema (backward compatible)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                id_column SERIAL PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                base_query TEXT,
                variant TEXT,
                frequency INTEGER DEFAULT 1
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id_column SERIAL PRIMARY KEY,
                cache_key TEXT NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY (cache_key) REFERENCES cache_entries(key) ON DELETE CASCADE
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query ON cache_entries(query)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_variant ON cache_entries(variant)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_base_query ON cache_entries(base_query)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_frequency ON cache_entries(frequency DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_variant_frequency ON cache_entries(variant, frequency DESC)')

        conn.commit()
        cursor.close()
        return conn

    def generate_key(self, query: str, variant: str = 'normal') -> str:
        """Generate a deterministic short hash key for a query with variant support."""
        base_query = query.replace('_detailed', '') if variant == 'detailed' else query
        return hashlib.md5(f"{base_query}:{variant}".encode('utf-8')).hexdigest()[:10]

    def add_entry(self, query: str, response: str, sources: List[str], variant: str = 'normal') -> str:
        """Add or update a query-response entry with variant and frequency support."""
        base_query = query.replace('_detailed', '') if variant == 'detailed' else query
        key = self.generate_key(base_query, variant)
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO cache_entries (key, query, base_query, variant, response, frequency, created_at)
                VALUES (%s, %s, %s, %s, %s, 1, %s)
                ON CONFLICT (key)
                DO UPDATE SET 
                    response = EXCLUDED.response, 
                    frequency = cache_entries.frequency + 1
            ''', (key, query, base_query, variant, response, datetime.now()))
            
            # Update sources
            cursor.execute("DELETE FROM sources WHERE cache_key = %s", (key,))
            for src in sources:
                cursor.execute("INSERT INTO sources (cache_key, source) VALUES (%s, %s)", (key, src))
            
            self.conn.commit()
            cursor.close()
        except Exception as e:
            self.conn.rollback()
            print(f"[Cache Error] Failed to add entry: {e}")
        
        return key

    def calculate_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_cached_response(self, query: str, query_embedding: List[float], embedding_model, variant: str = 'normal', skip_frequency_increment: bool = False) -> Optional[Dict[str, Any]]:
        """Get cached response with variant filtering and frequency boost."""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT key, base_query, response, frequency 
            FROM cache_entries 
            WHERE variant = %s 
            ORDER BY frequency DESC, created_at DESC
        """, (variant,))
        entries = cursor.fetchall()
        
        if not entries:
            cursor.close()
            return None
        
        # Semantic similarity with frequency boost
        query_embedding = np.array(query_embedding)
        best_score = -1
        best_entry = None
        
        for key, cached_base_query, response, frequency in entries:
            try:
                cached_embedding = embedding_model.get_text_embedding(cached_base_query)
                similarity = self.calculate_cosine_similarity(query_embedding, np.array(cached_embedding))
                
                # Boost score by frequency (popular items get preference)
                frequency_boost = min(frequency * 0.01, 0.1)  # Max 10% boost
                adjusted_score = similarity + frequency_boost
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_entry = {
                        "key": key,
                        "query": cached_base_query,
                        "response": response,
                        "similarity": similarity,
                        "frequency": frequency
                    }
            except Exception as e:
                print(f"[Cache Error] Failed to process entry {key}: {e}")
                continue
        
        cursor.close()
        
        if best_entry and best_entry["similarity"] >= self.similarity_threshold:
            # Increment frequency on cache hit (unless skipping)
            if not skip_frequency_increment:
                self._increment_frequency(best_entry["key"])
            return {
                "response": best_entry["response"],
                "original_query": best_entry["query"],
                "key": best_entry["key"],
                "similarity_score": best_entry["similarity"],
                "frequency": best_entry["frequency"],
                "is_cache_hit": True
            }
        
        return None

    def _increment_frequency(self, key: str):
        """Increment frequency counter when cache is hit."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE cache_entries SET frequency = frequency + 1 WHERE key = %s", (key,))
            self.conn.commit()
            cursor.close()
        except Exception as e:
            print(f"[Cache Error] Failed to increment frequency: {e}")

    def find_similar_entry(self, query: str, query_embedding: List[float], embedding_model) -> Tuple[Optional[Dict[str, Any]], float]:
        """Find a semantically similar query in cache using cosine similarity (legacy method for backward compatibility)."""
        return self._find_similar_entry_internal(query, query_embedding, embedding_model, variant='normal')

    def _find_similar_entry_internal(self, query: str, query_embedding: List[float], embedding_model, variant: str = 'normal') -> Tuple[Optional[Dict[str, Any]], float]:
        """Internal method to find similar entries with variant filtering."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, base_query, response, frequency FROM cache_entries WHERE variant = %s", (variant,))
        entries = cursor.fetchall()

        if not entries:
            cursor.close()
            return None, 0.0

        query_embedding = np.array(query_embedding)
        best_similarity = -1
        best_entry = None

        for key, cached_base_query, response, frequency in entries:
            try:
                cached_embedding = embedding_model.get_text_embedding(cached_base_query)
                similarity = self.calculate_cosine_similarity(query_embedding, np.array(cached_embedding))

                if similarity > best_similarity:
                    cursor.execute("SELECT source FROM sources WHERE cache_key = %s", (key,))
                    sources = [row[0] for row in cursor.fetchall()]
                    best_similarity = similarity
                    best_entry = {
                        "key": key,
                        "query": cached_base_query,
                        "response": response,
                        "frequency": frequency,
                        "sources": sources
                    }
            except Exception:
                continue

        cursor.close()

        if best_similarity >= self.similarity_threshold:
            return best_entry, best_similarity
        else:
            return None, 0.0

    def get_popular_queries(self, variant: str = 'normal', limit: int = 10) -> List[Dict]:
        """Get most popular queries for analytics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT base_query, frequency, created_at 
            FROM cache_entries 
            WHERE variant = %s 
            ORDER BY frequency DESC 
            LIMIT %s
        """, (variant, limit))
        
        results = []
        for base_query, frequency, created_at in cursor.fetchall():
            results.append({
                "query": base_query,
                "frequency": frequency,
                "created_at": created_at
            })
        
        cursor.close()
        return results

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                variant,
                COUNT(*) as total_entries,
                SUM(frequency) as total_hits,
                AVG(frequency) as avg_frequency,
                MAX(frequency) as max_frequency
            FROM cache_entries 
            GROUP BY variant
        """)
        
        stats = {}
        for variant, total_entries, total_hits, avg_frequency, max_frequency in cursor.fetchall():
            stats[variant] = {
                "total_entries": total_entries,
                "total_hits": total_hits,
                "avg_frequency": round(avg_frequency, 2) if avg_frequency else 0,
                "max_frequency": max_frequency
            }
        
        cursor.close()
        return stats

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def clear_cache(self) -> Dict[str, Any]:
        """Clear all cache entries from the database."""
        try:
            cursor = self.conn.cursor()
            
            # Clear sources table first due to foreign key constraint
            cursor.execute("DELETE FROM sources")
            sources_count = cursor.rowcount
            
            # Clear cache entries table
            cursor.execute("DELETE FROM cache_entries")
            entries_count = cursor.rowcount
            
            self.conn.commit()
            cursor.close()
            
            return {
                "success": True,
                "message": f"Cache cleared: {entries_count} entries and {sources_count} sources removed",
                "entries_cleared": entries_count,
                "sources_cleared": sources_count
            }
        except Exception as e:
            self.conn.rollback()
            return {
                "success": False,
                "message": f"Failed to clear cache: {str(e)}",
                "entries_cleared": 0,
                "sources_cleared": 0
            }
