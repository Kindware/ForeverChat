async function checkHealth() {
	try {
		const res = await fetch('/api/health');
		const data = await res.json();
		const badge = document.getElementById('health-badge');
		if (data.ok) {
			badge.classList.remove('d-none');
		}
	} catch (_) {}
}

async function loadModels() {
	const select = document.getElementById('model-select');
	if (!select) return;
	try {
		const res = await fetch('/api/models');
		const data = await res.json();
		select.innerHTML = '';
		const models = data.models || [];
		const def = data.default || '';
		if (!models.length) {
			const opt = document.createElement('option');
			opt.value = '';
			opt.textContent = 'No models found';
			select.appendChild(opt);
			select.disabled = true;
			return;
		}
		models.forEach(name => {
			const opt = document.createElement('option');
			opt.value = name;
			opt.textContent = name;
			if (name === def) opt.selected = true;
			select.appendChild(opt);
		});
	} catch (_) {
		select.innerHTML = '';
		const opt = document.createElement('option');
		opt.value = '';
		opt.textContent = 'Error loading models';
		select.appendChild(opt);
		select.disabled = true;
	}
}

function el(tag, className, text) {
	const e = document.createElement(tag);
	if (className) e.className = className;
	if (text !== undefined) e.textContent = text;
	return e;
}

function renderMessage(role, content) {
	const log = document.getElementById('chat-log');
	const row = el('div', `chat-message ${role} fade-in`);
	const bubble = el('div', 'bubble');
	bubble.textContent = content;
	row.appendChild(bubble);
	log.appendChild(row);
	log.scrollTop = log.scrollHeight;
}

function renderTopics(topics) {
	const list = document.getElementById('topic-list');
	list.innerHTML = '';
	if (!topics || topics.length === 0) {
		list.innerHTML = '<li class="text-secondary">No active topics yet.</li>';
		return;
	}
	topics.forEach(t => {
		const li = el('li', 'mb-2');
		const header = el('div', 'd-flex justify-content-between align-items-center');
		const left = el('div', 'small');
		const share = typeof t.share_percent === 'number' ? `${t.share_percent}%` : '';
		left.innerHTML = `<span class="me-1">${t.status || '•'}</span> <strong>${t.name}</strong> — ${t.exchanges} exchanges${share ? ` (${share})` : ''}`;
		const btn = el('button', 'btn btn-outline-danger btn-sm');
		btn.type = 'button';
		btn.textContent = 'Archive';
		btn.setAttribute('data-forget-topic', t.name);
		header.appendChild(left);
		header.appendChild(btn);
		li.appendChild(header);

		if (typeof t.share_percent === 'number') {
			const barOuter = el('div', 'progress mt-1');
			const barInner = el('div', 'progress-bar bg-primary');
			barInner.style.width = `${Math.min(t.share_percent, 100)}%`;
			barInner.setAttribute('aria-valuenow', t.share_percent);
			barInner.setAttribute('aria-valuemin', '0');
			barInner.setAttribute('aria-valuemax', '100');
			barOuter.appendChild(barInner);
			li.appendChild(barOuter);
		}

		list.appendChild(li);
	});
}

function updateStats(stats) {
	if (!stats) return;
	const tokenEl = document.getElementById('token-stats');
	tokenEl.textContent = `${stats.current_tokens} / ${stats.max_tokens}`;
}

async function sendMessage(message) {
	const form = document.getElementById('chat-form');
	const input = document.getElementById('chat-input');
	const modelSelect = document.getElementById('model-select');
	const model = modelSelect && modelSelect.value ? modelSelect.value : undefined;
	input.disabled = true;
	form.querySelector('button[type="submit"]').disabled = true;

	try {
		renderMessage('user', message);
		const thinkingId = 'thinking-' + Date.now();
		renderMessage('assistant thinking', 'Thinking…');

		const body = { message };
		if (model) body.model = model;

		const res = await fetch('/api/chat', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
		const data = await res.json();

		// Replace the last "Thinking…" bubble with real reply
		const log = document.getElementById('chat-log');
		const last = log.querySelector('.chat-message.assistant.thinking');
		if (last) last.remove();
		renderMessage('assistant', data.reply || '[No response]');

		updateStats(data.stats);
		renderTopics(data.topics);
	} catch (err) {
		const log = document.getElementById('chat-log');
		const last = log.querySelector('.chat-message.assistant.thinking');
		if (last) last.remove();
		renderMessage('system', 'Error sending message. Please try again.');
	} finally {
		input.disabled = false;
		form.querySelector('button[type="submit"]').disabled = false;
		input.focus();
	}
}

async function sendCommand(command) {
	try {
		const res = await fetch('/api/command', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ command })
		});
		const data = await res.json();
		if (data.error) {
			renderMessage('system', data.error);
			return;
		}
		if (data.message) renderMessage('system', data.message);
		if (data.topics) renderTopics(data.topics);
		if (data.stats) updateStats(data.stats);
	} catch (_) {
		renderMessage('system', 'Command failed.');
	}
}

document.addEventListener('DOMContentLoaded', () => {
	checkHealth();
	loadModels();

	const form = document.getElementById('chat-form');
	const input = document.getElementById('chat-input');
	form.addEventListener('submit', (e) => {
		e.preventDefault();
		const msg = input.value.trim();
		if (!msg) return;
		input.value = '';
		if (msg.startsWith('/')) {
			sendCommand(msg);
		} else {
			sendMessage(msg);
		}
	});

	document.getElementById('clear-btn').addEventListener('click', (e) => {
		e.preventDefault();
		sendCommand('/clear');
		const log = document.getElementById('chat-log');
		log.innerHTML = '';
		renderMessage('system', 'Conversation cleared.');
	});

	document.querySelectorAll('[data-command]').forEach(btn => {
		btn.addEventListener('click', () => {
			const cmd = btn.getAttribute('data-command');
			sendCommand(cmd);
		});
	});

	// Topic list: archive individual topics
	document.getElementById('topic-list').addEventListener('click', (e) => {
		const btn = e.target.closest('[data-forget-topic]');
		if (!btn) return;
		const topic = btn.getAttribute('data-forget-topic');
		if (!topic) return;
		sendCommand(`/forget ${topic}`);
	});
});


