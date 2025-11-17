#!/usr/bin/env python3
import os
from uuid import uuid4
from typing import Dict, List
from flask import Flask, render_template, request, jsonify, session
from smart_chat_app import SmartContextManager, OllamaClient, MODEL_CONTEXT_CONFIG, DEFAULT_MAX_TOKENS
from rag_store import rag_store


def create_app() -> Flask:
	app = Flask(__name__, static_folder="static", template_folder="templates")
	app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

	# Per-session context managers, ephemeral in-memory
	session_id_to_context: Dict[str, SmartContextManager] = {}

	# Default model and base URL for creating clients per request
	default_model = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
	default_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
	# Reference client for getting base_url and listing models
	reference_client = OllamaClient(model=default_model, base_url=default_base_url)

	def get_max_tokens_for_model(model_name: str) -> int:
		"""Return max token budget to use for a given model."""
		return MODEL_CONTEXT_CONFIG.get(model_name, DEFAULT_MAX_TOKENS)

	def get_session_id() -> str:
		if "sid" not in session:
			session["sid"] = str(uuid4())
		return session["sid"]

	def get_context_manager() -> SmartContextManager:
		sid = get_session_id()
		if sid not in session_id_to_context:
			session_id_to_context[sid] = SmartContextManager(max_tokens=get_max_tokens_for_model(default_model))
		return session_id_to_context[sid]

	@app.get("/api/models")
	def models():
		"""List available Ollama models."""
		models = reference_client.list_models()
		return jsonify({"models": models, "default": default_model})

	@app.get("/")
	def index():
		return render_template("index.html")

	@app.get("/api/health")
	def health():
		is_ok = reference_client.test_connection()
		return jsonify({"ok": is_ok, "model": reference_client.model})

	@app.post("/api/chat")
	def chat():
		data = request.get_json(silent=True) or {}
		user_message = (data.get("message") or "").strip()
		model = (data.get("model") or default_model).strip() or default_model

		max_tokens = get_max_tokens_for_model(model)

		if not user_message:
			return jsonify({"error": "Message is required"}), 400

		# Always create a fresh client for each request to avoid any state issues
		# This ensures each request gets a clean client instance
		client = OllamaClient(model=model, base_url=default_base_url)

		context_manager = get_context_manager()

		# Keep context manager aligned with chosen model's context window.
		if context_manager.max_tokens != max_tokens:
			context_manager.max_tokens = max_tokens
			context_manager.active_budget = int(max_tokens * 0.5)

		# Prepare messages with existing context
		messages = context_manager.get_context_for_llm()

		# RAG: retrieve semantically similar long-term exchanges (scoped by active topics)
		active_topics: List[Dict] = context_manager.get_active_topics()
		active_topic_names = [t["name"] for t in active_topics]
		retrieved = rag_store.search(user_message, topics=active_topic_names, k=4)
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

		messages.append({"role": "user", "content": user_message})

		assistant_response = client.chat(messages, num_ctx=max_tokens)

		# Store exchange in short-term and long-term memory
		context_manager.add_exchange(user_message, assistant_response)

		# Derive topics again for RAG metadata (cheap and keeps RAG decoupled)
		topics_for_rag = context_manager.topic_detector.detect_topics(
			user_message + " " + assistant_response,
			user_text=user_message,
		)
		try:
			rag_store.add_exchange(
				user_message=user_message,
				assistant_response=assistant_response,
				topics=topics_for_rag,
				model=model,
			)
		except Exception:
			# RAG failures should never break the core chat flow.
			pass

		return jsonify({
			"reply": assistant_response,
			"stats": {
				"total_exchanges": context_manager.stats["total_exchanges"],
				"current_tokens": context_manager.current_tokens,
				"max_tokens": context_manager.max_tokens
			},
			"topics": context_manager.get_active_topics()
		})

	@app.post("/api/command")
	def command():
		data = request.get_json(silent=True) or {}
		cmd = (data.get("command") or "").strip()
		if not cmd:
			return jsonify({"error": "Command is required"}), 400

		context_manager = get_context_manager()
		# Limited subset of commands for safety in web UI
		if cmd.startswith("/forget "):
			topic = cmd.split(maxsplit=1)[1]
			context_manager.forget_topic(topic)
			return jsonify({"ok": True, "message": f"Topic '{topic}' archived."})
		if cmd == "/clear":
			context_manager.active_exchanges = []
			context_manager.current_tokens = 0
			return jsonify({"ok": True, "message": "Conversation cleared."})
		if cmd == "/topics":
			topics = context_manager.get_active_topics()
			if topics:
				msg = f"{len(topics)} active topic(s): " + ", ".join(t["name"] for t in topics)
			else:
				msg = "No active topics."
			return jsonify({"ok": True, "topics": topics, "message": msg})
		if cmd == "/stats":
			stats = context_manager.stats
			msg = (
				f"Total exchanges: {stats['total_exchanges']}, "
				f"active exchanges: {len(context_manager.active_exchanges)}, "
				f"tokens: {context_manager.current_tokens}/{context_manager.max_tokens}, "
				f"topics archived: {stats['topics_archived']}, "
				f"topics retrieved: {stats['topics_retrieved']}."
			)
			return jsonify({
				"ok": True,
				"message": msg,
				"stats": {
					"total_exchanges": stats["total_exchanges"],
					"active_exchanges": len(context_manager.active_exchanges),
					"current_tokens": context_manager.current_tokens,
					"max_tokens": context_manager.max_tokens,
					"chars_saved": stats["chars_saved"],
					"topics_archived": stats["topics_archived"],
					"topics_retrieved": stats["topics_retrieved"],
				}
			})

		return jsonify({"error": "Unknown command"}), 400

	return app


if __name__ == "__main__":
	app = create_app()
	port = int(os.environ.get("PORT", "5000"))
	app.run(host="0.0.0.0", port=port, debug=True)


