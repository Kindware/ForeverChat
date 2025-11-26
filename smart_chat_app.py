#!/usr/bin/env python3
"""
Smart Chat Application with Advanced Context Management

Features:
- Topic-aware context management (auto-juggling)
- Semantic deduplication (catches similar phrases)
- Automatic cleaning (removes fluff)
- User commands (forget topics, show stats)
- Works with local Ollama (Gemma 3:12b)

Usage:
    python smart_chat_app.py
    
Commands:
    /forget <topic>   - Archive a topic
    /topics           - Show active topics
    /stats            - Show context statistics
    /clear            - Clear conversation
    /help             - Show help
    /quit             - Exit
"""

import os
import json
import time
import hashlib
import re
import requests
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from rag_store import rag_store

# ============================================================================
# Model / Context Configuration
# ============================================================================

# Default maximum context tokens when we don't have a specific model entry.
DEFAULT_MAX_TOKENS = 4000

# Per-model context window hints (num_ctx) so we can size our context manager
# and Ollama requests appropriately. Adjust these to match your local models.
MODEL_CONTEXT_CONFIG: Dict[str, int] = {
    # Gemma examples - using reasonable context sizes to avoid freezing
    "gemma3:12b": 8192,
    # Llama examples - using reasonable context sizes to avoid freezing
    "llama3.2:3b": 8192,
    "llama3.1:8b": 8192,
    # Pervert examples LOL
    "HammerAI/mythomax-l2": 8000,
}

# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class TopicTag:
    """Represents a conversation topic"""
    name: str
    keywords: Set[str]
    created_at: str
    last_mentioned: str
    mention_count: int = 0
    is_active: bool = True
    
    def update_mention(self):
        self.last_mentioned = datetime.now().isoformat()
        self.mention_count += 1

@dataclass
class ChatExchange:
    """A single chat exchange with topic awareness"""
    timestamp: str
    user_message: str
    assistant_response: str
    tokens_used: int
    priority_score: float
    topics: List[str]
    exchange_id: str
    is_cleaned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Context Cleaning
# ============================================================================

class ContextCleaner:
    """Removes conversational fluff while preserving meaning"""
    
    FLUFF_PATTERNS = [
        r'\b(um+|uh+|hmm+|err+|ah+)\b',
        r'\b(like|you know|I mean|basically|actually)\b(?=\s)',
        r'(?:^|\.\s+)(well|so|anyway|anyways),?\s+',
        r'\s+(lol|haha|hehe)\s+',
    ]
    
    SENTIMENT_ONLY = [
        r"(?i)^(thanks?|thank you|thx|ty)!*$",
        r"(?i)^(that'?s? (?:great|awesome|perfect|amazing|cool|nice))!*$",
        r"(?i)^(ok+ay?|alright|sounds good|perfect|got it)!*$",
        r"(?i)^(yes|yeah|yep|yup|sure|absolutely|definitely)!*$",
    ]
    
    @staticmethod
    def clean_message(text: str) -> Tuple[str, Dict[str, Any]]:
        """Clean a message, return (cleaned_text, metadata)"""
        original_length = len(text)
        metadata = {'was_cleaned': False, 'sentiment': None}
        
        # Check if pure sentiment
        for pattern in ContextCleaner.SENTIMENT_ONLY:
            if re.match(pattern, text.strip()):
                metadata['sentiment'] = 'positive'
                return text, metadata
        
        cleaned = text
        
        # Remove fluff
        for pattern in ContextCleaner.FLUFF_PATTERNS:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Clean whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if len(cleaned) < original_length * 0.95:
            metadata['was_cleaned'] = True
            metadata['chars_saved'] = original_length - len(cleaned)
        
        return cleaned, metadata

# ============================================================================
# Topic Detection
# ============================================================================

class TopicDetector:
    """Detects and tracks topics in conversation"""
    
    TOPIC_PATTERNS = {
        'programming': {
            'keywords': ['code', 'programming', 'python', 'javascript', 'react', 'api', 'function', 'bug', 'debug', 'develop'],
        },
        'ai': {
            'keywords': ['ai', 'llm', 'gpt', 'model', 'training', 'prompt', 'context', 'token', 'neural', 'machine learning'],
        },
        'homelab': {
            'keywords': ['server', 'proxmox', 'docker', 'container', 'vm', 'virtual machine', 'nas', 'network', 'selfhost'],
        },
        'hardware': {
            'keywords': ['cpu', 'gpu', 'ram', 'storage', 'disk', 'motherboard', 'nvidia', 'amd', 'intel'],
        },
        'linux': {
            'keywords': ['linux', 'ubuntu', 'debian', 'arch', 'terminal', 'bash', 'shell', 'command'],
        },
    }
    
    # Simple stopwords for dynamic topic extraction
    STOPWORDS = {
        'the','a','an','and','or','but','if','then','else','when','while','for','to','of','in','on','at','by','with',
        'from','as','is','are','was','were','be','been','being','it','its','this','that','these','those','i','you',
        'we','they','he','she','them','him','her','my','your','our','their','mine','yours','ours','theirs','me',
        'do','did','does','doing','have','has','had','having','can','could','should','would','may','might','will',
        'shall','about','into','over','under','again','further','then','once','here','there','why','how','what','who',
        'which','because','so','such','very','just','not','no','yes','ok','okay','hi','hey','hello','thanks','thank',
        'please','like','really','actually','basically','literally'
    }
    
    # Words we never create topics for
    GENERIC_EXCLUSIONS = {
        'topic','issue','question','thing','stuff','idea','point','context','message','chat',
        # Generic-ish content words that showed up as low-value topics
        'elements','data','incorporated','russian','computer','lab','environment'
    }
    
    def __init__(self):
        self.topics: Dict[str, TopicTag] = {}
        self._initialize_topics()
        # Config for dynamic topic creation
        self.enable_dynamic_topics: bool = True
        # Allow shorter, meaningful words like "cat" while still filtering noise
        self.min_keyword_length: int = 3
        # Prefer a single strong new topic per exchange over many weak ones
        self.max_dynamic_per_message: int = 1
    
    def _initialize_topics(self):
        for topic_name, info in self.TOPIC_PATTERNS.items():
            self.topics[topic_name] = TopicTag(
                name=topic_name,
                keywords=set(info['keywords']),
                created_at=datetime.now().isoformat(),
                last_mentioned=datetime.now().isoformat(),
            )
    
    def detect_topics(self, text: str, user_text: Optional[str] = None) -> List[str]:
        """Detect topics in text.

        user_text (optional) lets us boost terms that appear in the user's
        question/statement relative to words that only appear in the assistant
        explanation.
        """
        text_lower = text.lower()
        detected = []
        
        for topic_name, topic in self.topics.items():
            if any(keyword in text_lower for keyword in topic.keywords):
                detected.append(topic_name)
                topic.update_mention()
        
        # Dynamic topic creation: add new topics based on keywords not covered above
        if self.enable_dynamic_topics:
            new_topics = self._detect_and_create_dynamic_topics(text, user_text=user_text)
            for nt in new_topics:
                if nt not in detected:
                    detected.append(nt)
        
        return detected
    
    def add_custom_topic(self, name: str, keywords: List[str]):
        """Add a user-defined topic"""
        self.topics[name] = TopicTag(
            name=name,
            keywords=set(keywords),
            created_at=datetime.now().isoformat(),
            last_mentioned=datetime.now().isoformat(),
        )
    
    def _detect_and_create_dynamic_topics(self, original_text: str, user_text: Optional[str] = None) -> List[str]:
        """
        Heuristic extraction of candidate keywords from text and creation of topics for them.
        - Prefers nouns/proper-looking words by simple casing/length rules (no heavy NLP deps).
        - Skips stopwords and generic terms.
        - Generates small keyword sets (singular/plural variants).
        """
        candidates = self._extract_candidate_keywords(original_text, user_text=user_text)
        created: List[str] = []
        for word in candidates:
            if len(created) >= self.max_dynamic_per_message:
                break
            if word in self.topics:
                continue
            # Avoid creating dynamic topic if word is already a keyword of any topic
            if any(word in t.keywords for t in self.topics.values()):
                continue
            self._create_dynamic_topic(word)
            created.append(word)
        return created
    
    def _extract_candidate_keywords(self, text: str, user_text: Optional[str] = None) -> List[str]:
        """
        Very lightweight keyword extractor:
        - Tokenize by non-letters
        - Filter by length, stopwords, exclusions
        - Prefer capitalized terms (proper nouns) and rare-ish words
        - Return at most a few candidates ranked by simple scoring
        """
        # Keep a copy with original casing to detect Proper Nouns
        original_tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        lower_tokens = [t.lower() for t in original_tokens]

        # Tokens from the user portion of the exchange so we can boost them.
        user_tokens_lower: Set[str] = set()
        if user_text:
            user_tokens_lower = {
                t.lower()
                for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", user_text)
            }
        
        counts: Dict[str, int] = defaultdict(int)
        proper_like: Set[str] = set()
        for orig, low in zip(original_tokens, lower_tokens):
            if len(low) < self.min_keyword_length:
                continue
            if low in self.STOPWORDS or low in self.GENERIC_EXCLUSIONS:
                continue
            # Base frequency count
            counts[low] += 1

            # Boost tokens that appear in the user question/statement.
            if low in user_tokens_lower:
                counts[low] += 1
            if orig[:1].isupper() and orig[1:].islower():
                proper_like.add(low)
        
        if not counts:
            return []
        
        # Simple score: frequency + proper-like bonus, penalize if substring of existing topic keywords
        def score(term: str) -> float:
            s = counts[term]
            if term in proper_like:
                s += 0.5
            if any(term in kw for t in self.topics.values() for kw in t.keywords):
                s -= 0.25
            return s
        
        ranked = sorted(counts.keys(), key=lambda w: (-score(w), w))
        # Return a small set
        return ranked[: self.max_dynamic_per_message]
    
    def _create_dynamic_topic(self, name: str):
        """Create a topic with simple variant keywords (singular/plural)."""
        variants = self._variants(name)
        self.topics[name] = TopicTag(
            name=name,
            keywords=variants,
            created_at=datetime.now().isoformat(),
            last_mentioned=datetime.now().isoformat(),
        )
        # Touch mention to mark activity
        self.topics[name].update_mention()
    
    def _variants(self, word: str) -> Set[str]:
        """
        Generate naive singular/plural variants.
        This avoids external NLP dependencies while covering common cases.
        """
        forms: Set[str] = {word}
        w = word
        # plural to singular
        if w.endswith('ies') and len(w) > 3:
            forms.add(w[:-3] + 'y')
        if w.endswith('es') and len(w) > 2:
            forms.add(w[:-2])
        if w.endswith('s') and len(w) > 1 and not w.endswith('ss'):
            forms.add(w[:-1])
        # singular to plural
        if w.endswith('y') and len(w) > 1:
            forms.add(w[:-1] + 'ies')
        forms.add(w + 's')
        forms.add(w + 'es')
        return set(f for f in forms if len(f) >= self.min_keyword_length)

# ============================================================================
# Topic-Aware Context Manager
# ============================================================================

class SmartContextManager:
    """
    Advanced context manager with:
    - Topic tracking and auto-juggling
    - Automatic cleaning
    - Smart archival
    """
    
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.max_tokens = max_tokens
        # Active context is capped to 50% of the total budget so we always
        # have free room available for retrieving archived topics.
        self.active_budget = int(max_tokens * 0.5)
        self.current_tokens = 0
        
        # Exchanges
        self.active_exchanges: List[ChatExchange] = []
        self.archived_by_topic: Dict[str, List[ChatExchange]] = defaultdict(list)
        
        # Components
        self.cleaner = ContextCleaner()
        self.topic_detector = TopicDetector()
        
        # Topic tracking
        self.exchanges_since_topic_mention: Dict[str, int] = defaultdict(int)
        self.topic_juggle_threshold = 10  # Archive after 10 exchanges without mention
        
        # Stats
        self.stats = {
            'total_exchanges': 0,
            'chars_saved': 0,
            'topics_archived': 0,
            'topics_retrieved': 0,
        }
    
    def add_exchange(self, user_message: str, assistant_response: str) -> str:
        """Add exchange with automatic topic management"""
        
        # Clean messages
        user_cleaned, user_meta = self.cleaner.clean_message(user_message)
        assistant_cleaned, assistant_meta = self.cleaner.clean_message(assistant_response)
        
        # Update stats
        self.stats['chars_saved'] += user_meta.get('chars_saved', 0) + assistant_meta.get('chars_saved', 0)
        
        # Detect topics (give detector both full text and the raw user text so
        # it can prioritise the user's main nouns as topics).
        topics = self.topic_detector.detect_topics(
            user_cleaned + " " + assistant_cleaned,
            user_text=user_cleaned,
        )
        
        # Create exchange
        exchange = ChatExchange(
            timestamp=datetime.now().isoformat(),
            user_message=user_cleaned,
            assistant_response=assistant_cleaned,
            tokens_used=self._estimate_tokens(user_cleaned + assistant_cleaned),
            priority_score=self._calculate_priority(user_cleaned, assistant_cleaned),
            topics=topics,
            exchange_id=self._generate_id(),
            is_cleaned=True,
            metadata={'user_meta': user_meta, 'assistant_meta': assistant_meta}
        )
        
        # Add to active
        self.active_exchanges.append(exchange)
        self.current_tokens += exchange.tokens_used
        self.stats['total_exchanges'] += 1
        
        # Update topic mentions
        self._update_topic_mentions(topics)
        
        # Check for topic retrieval
        self._check_topic_retrieval(topics)
        
        # Auto-juggle cold topics
        self._auto_juggle_topics()
        
        # Enforce active budget so at least half the window stays free
        self._enforce_active_budget()
        
        return exchange.exchange_id
    
    def _update_topic_mentions(self, mentioned_topics: List[str]):
        """Track which topics are getting cold"""
        for topic in self.topic_detector.topics.keys():
            if topic not in mentioned_topics:
                self.exchanges_since_topic_mention[topic] += 1
            else:
                self.exchanges_since_topic_mention[topic] = 0
    
    def _check_topic_retrieval(self, mentioned_topics: List[str]):
        """Retrieve archived topics if mentioned"""
        for topic in mentioned_topics:
            if topic in self.archived_by_topic and self.archived_by_topic[topic]:
                self._retrieve_topic(topic)
    
    def _retrieve_topic(self, topic_name: str):
        """Bring archived topic back to active context.
        
        Retrieves as many exchanges as can fit within the active_budget,
        prioritizing the most recent exchanges.
        """
        archived = self.archived_by_topic[topic_name]
        if not archived:
            return
        
        # Sort by timestamp (most recent first) to prioritize recent exchanges
        sorted_archived = sorted(archived, key=lambda x: x.timestamp, reverse=True)
        
        retrieved_count = 0
        remaining_archived = []
        
        # Retrieve exchanges until we would exceed the active_budget
        for exchange in sorted_archived:
            # Check if adding this exchange would exceed the active budget
            if self.current_tokens + exchange.tokens_used > self.active_budget:
                # Keep this and remaining exchanges in archived
                remaining_archived.append(exchange)
            else:
                # This exchange fits, add it back to active context
                self.active_exchanges.append(exchange)
                self.current_tokens += exchange.tokens_used
                retrieved_count += 1
        
        # Update archived list with exchanges that didn't fit
        self.archived_by_topic[topic_name] = remaining_archived
        
        if retrieved_count > 0:
            print(f"Retrieving archived topic '{topic_name}': {retrieved_count} exchange(s) "
                  f"({len(remaining_archived)} still archived)")
            self.stats['topics_retrieved'] += 1

        # After retrieval, enforce budget to ensure we're still within limits
        # (This is a safety check in case of edge cases)
        self._enforce_active_budget()
    
    def _auto_juggle_topics(self):
        """Archive topics that haven't been mentioned recently"""
        cold_topics = [
            topic for topic, count in self.exchanges_since_topic_mention.items()
            if count >= self.topic_juggle_threshold
        ]
        
        for topic in cold_topics:
            self._archive_topic(topic, silent=True)
    
    def _archive_topic(self, topic_name: str, silent: bool = False):
        """Archive all exchanges for a topic"""
        to_archive = [ex for ex in self.active_exchanges if topic_name in ex.topics]
        
        if not to_archive:
            return
        
        if not silent:
            print(f"Archiving topic '{topic_name}' ({len(to_archive)} exchanges)")
        
        for exchange in to_archive:
            self.archived_by_topic[topic_name].append(exchange)
            self.active_exchanges.remove(exchange)
            self.current_tokens -= exchange.tokens_used
        
        self.stats['topics_archived'] += 1
    
    def _archive_by_priority(self):
        """Backwards-compatible alias for budget enforcement."""
        self._enforce_active_budget()

    def _enforce_active_budget(self):
        """
        Ensure active context never exceeds the configured active_budget.
        We always keep at least ~50% of the max token window empty so there is
        room to retrieve archived topics without overflowing.
        """
        if not self.active_exchanges:
            return

        # If we're within budget, nothing to do.
        if self.current_tokens <= self.active_budget:
            return

        # Sort by (priority_score asc, timestamp asc) so we archive the least
        # important and oldest exchanges first.
        sorted_exchanges = sorted(
            self.active_exchanges,
            key=lambda x: (x.priority_score, x.timestamp)
        )

        for exchange in sorted_exchanges:
            if self.current_tokens <= self.active_budget:
                break

            # Send exchange to archived buckets by topic
            if exchange.topics:
                for topic in exchange.topics:
                    self.archived_by_topic[topic].append(exchange)
            else:
                self.archived_by_topic['general'].append(exchange)

            if exchange in self.active_exchanges:
                self.active_exchanges.remove(exchange)
                self.current_tokens -= exchange.tokens_used
    
    def forget_topic(self, topic_name: str):
        """User command: forget about a topic"""
        self._archive_topic(topic_name, silent=False)
        self.exchanges_since_topic_mention[topic_name] = 0
    
    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """Format active context for LLM"""
        messages = []
        for ex in sorted(self.active_exchanges, key=lambda x: x.timestamp):
            messages.append({'role': 'user', 'content': ex.user_message})
            messages.append({'role': 'assistant', 'content': ex.assistant_response})
        return messages
    
    def get_active_topics(self) -> List[Dict[str, Any]]:
        """Get info about active topics"""
        topics: List[Dict[str, Any]] = []
        total_exchanges = max(self.stats.get('total_exchanges', 0), 1)
        for topic_name, topic in self.topic_detector.topics.items():
            count = sum(1 for ex in self.active_exchanges if topic_name in ex.topics)
            if count > 0:
                share = round((count / total_exchanges) * 100, 1)
                topics.append({
                    'name': topic_name,
                    'exchanges': count,
                    'last_mentioned': self.exchanges_since_topic_mention[topic_name],
                    'status': 'XX' if self.exchanges_since_topic_mention[topic_name] < 3 else 'XXX',
                    'share_percent': share,
                })
        return sorted(topics, key=lambda x: x['last_mentioned'])
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return int(len(text.split()) * 1.3)
    
    def _calculate_priority(self, user_msg: str, assistant_msg: str) -> float:
        """Calculate exchange priority"""
        score = 0.5
        if '?' in user_msg:
            score += 0.3
        if any(word in (user_msg + assistant_msg).lower() for word in ['error', 'bug', 'help', 'how']):
            score += 0.2
        return min(score, 1.0)
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]

# ============================================================================
# Ollama Integration
# ============================================================================

class OllamaClient:
    """Client for local Ollama API"""
    
    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: Optional[int] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        # Allow overriding via env; fall back to a generous default for big models.
        env_timeout = os.environ.get("OLLAMA_HTTP_TIMEOUT")
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        elif env_timeout and env_timeout.isdigit():
            self.timeout_seconds = int(env_timeout)
        else:
            self.timeout_seconds = 300
    
    def chat(self, messages: List[Dict[str, str]], num_ctx: Optional[int] = None) -> str:
        """Send chat request to Ollama.

        num_ctx controls the context window size on the Ollama side.
        """
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
            if num_ctx is not None:
                payload["options"] = {"num_ctx": int(num_ctx)}

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            
            if response.status_code == 200:
                return response.json()['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Is it running? (ollama serve)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """Return a list of available model names from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return []
            data = response.json()
            # Ollama returns {"models":[{"name":"llama3","modified_at":...}, ...]}
            models = data.get("models", [])
            return [m.get("name") for m in models if m.get("name")]
        except Exception:
            return []

# ============================================================================
# Chat Application
# ============================================================================

class SmartChatApp:
    """Main chat application"""
    
    def __init__(self, model: str = "gemma3:12b"):
        # Size context window based on model if we have a hint
        max_tokens = MODEL_CONTEXT_CONFIG.get(model, DEFAULT_MAX_TOKENS)
        self.context_manager = SmartContextManager(max_tokens=max_tokens)
        self.ollama = OllamaClient(model=model)
        self.running = False
        
        print("="*70)
        print("FOREVERCHAT APPLICATION")
        print("="*70)
        print(f"Model: {model}")
        print("Features: Topic tracking, Auto-cleaning, Smart archival")
        print("\nCommands: /help, /topics, /stats, /forget <topic>, /clear, /quit")
        print("="*70)
    
    def start(self):
        """Start the chat loop"""
        
        # Test Ollama connection
        if not self.ollama.test_connection():
            print("Cannot connect to Ollama!")
            print("Please start Ollama: ollama serve")
            print(f"Then pull the model: ollama pull gemma3:12b")
            return
        
        print("Connected to Ollama\n")
        
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Get response from Ollama
                print("Assistant: ", end='', flush=True)
                
                # Prepare messages
                messages = self.context_manager.get_context_for_llm()

                # RAG: retrieve semantically similar long-term exchanges (scoped by active topics)
                active_topics = self.context_manager.get_active_topics()
                active_topic_names = [t["name"] for t in active_topics]
                retrieved = rag_store.search(user_input, topics=active_topic_names, k=4)
                if retrieved:
                    context_snippets = "\n\n".join(r["text"] for r in retrieved)
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": "Here are relevant notes from past conversations:\n\n"
                            + context_snippets,
                        },
                    )

                messages.append({'role': 'user', 'content': user_input})
                
                # Get response
                response = self.ollama.chat(messages, num_ctx=self.context_manager.max_tokens)
                print(response)
                
                # Add to context
                self.context_manager.add_exchange(user_input, response)

                # RAG: store exchange with topics and model metadata
                topics_for_rag = self.context_manager.topic_detector.detect_topics(
                    user_input + " " + response,
                    user_text=user_input,
                )
                try:
                    rag_store.add_exchange(
                        user_message=user_input,
                        assistant_response=response,
                        topics=topics_for_rag,
                        model=self.ollama.model,
                    )
                except Exception:
                    # RAG failures should not break CLI chat.
                    pass
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _handle_command(self, command: str):
        """Handle special commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            print("Commands:")
            print("  /topics          - Show active topics")
            print("  /stats           - Show statistics")
            print("  /forget <topic>  - Archive a topic")
            print("  /clear           - Clear conversation")
            print("  /quit            - Exit")
        
        elif cmd == '/topics':
            topics = self.context_manager.get_active_topics()
            if topics:
                print("Active Topics:")
                for t in topics:
                    print(f"  {t['status']} {t['name']}: {t['exchanges']} exchanges, "
                          f"last mentioned {t['last_mentioned']} exchanges ago")
            else:
                print("No active topics")
        
        elif cmd == '/stats':
            stats = self.context_manager.stats
            print(f"Statistics:")
            print(f"  Total exchanges: {stats['total_exchanges']}")
            print(f"  Active exchanges: {len(self.context_manager.active_exchanges)}")
            print(f"  Current tokens: {self.context_manager.current_tokens}/{self.context_manager.max_tokens}")
            print(f"  Characters saved: {stats['chars_saved']}")
            print(f"  Topics archived: {stats['topics_archived']}")
            print(f"  Topics retrieved: {stats['topics_retrieved']}")
        
        elif cmd == '/forget':
            if len(parts) < 2:
                print("Usage: /forget <topic>")
            else:
                topic = parts[1]
                self.context_manager.forget_topic(topic)
        
        elif cmd == '/clear':
            self.context_manager.active_exchanges = []
            self.context_manager.current_tokens = 0
            print("Conversation cleared")
        
        elif cmd == '/quit':
            self.running = False
            print("Goodbye!")
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import sys
    
    # Parse model from command line (optional)
    model = "gemma3:12b"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    # Create and start app
    app = SmartChatApp(model=model)
    app.start()

if __name__ == "__main__":
    main()
