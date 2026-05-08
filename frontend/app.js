// --- Optimized Markdown Renderer Configuration ---
const customRenderer = {
    code(code, language) {
        const validLang = language && hljs.getLanguage(language) ? language : 'plaintext';
        const highlighted = validLang === 'plaintext'
            ? hljs.highlightAuto(code).value
            : hljs.highlight(code, { language: validLang }).value;

        return `
<div class="code-block-wrapper">
    <div class="code-block-header">
        <span class="code-lang">${validLang}</span>
        <button class="copy-btn" aria-label="Copy code">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            Copy
        </button>
    </div>
    <pre><code class="hljs language-${validLang}">${highlighted}</code></pre>
</div>`;
    }
};

marked.use({ renderer: customRenderer });

marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false
});

// --- DOMPurify Configuration ---
const purifyConfig = {
    ALLOWED_TAGS: [
        'p', 'br', 'strong', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'code', 'pre', 'span', 'div', 'button', 'svg',
        'path', 'rect', 'polyline', 'circle', 'line', 'blockquote', 'em', 'i'
    ],
    ALLOWED_ATTR: [
        'class', 'id', 'xmlns', 'width', 'height', 'viewBox',
        'fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin',
        'x', 'y', 'rx', 'ry', 'd', 'points', 'cx', 'cy', 'r', 'x1', 'y1', 'x2', 'y2',
        'aria-label'
    ]
};

function safeRender(container, html) {
    container.innerHTML = DOMPurify.sanitize(html, purifyConfig);
}

// Event delegation for copy buttons
document.addEventListener('click', (e) => {
    const button = e.target.closest('.copy-btn');
    if (!button) return;

    const wrapper = button.closest('.code-block-wrapper');
    const codeEl = wrapper.querySelector('code');
    const text = codeEl.textContent;

    navigator.clipboard.writeText(text).then(() => {
        const originalHTML = button.innerHTML;
        button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
        setTimeout(() => button.innerHTML = originalHTML, 2000);
    });
});

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const themeToggle = document.getElementById('themeToggle');
const sunIcon = document.getElementById('sunIcon');
const moonIcon = document.getElementById('moonIcon');
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');

const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://kaustav2006-chatbot-api.hf.space';

const sessionId = Math.random().toString(36).substring(7);

// --- Theme Toggle Logic ---
const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeIcons(savedTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcons(newTheme);
});

function updateThemeIcons(theme) {
    sunIcon.style.display = theme === 'dark' ? 'block' : 'none';
    moonIcon.style.display = theme === 'dark' ? 'none' : 'block';
}

// --- Textarea Auto-resize ---
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
    sendButton.disabled = this.value.trim() === '';
});

// --- Chat Logic ---
function createMessageElement(sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'ai-message');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar', sender === 'user' ? 'user-avatar' : 'ai-avatar');
    if (sender === 'user') {
        avatar.textContent = 'U';
    } else {
        avatar.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>';
    }

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    return { messageDiv, contentDiv };
}

function showTypingIndicator(contentDiv) {
    safeRender(contentDiv, `
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `);
}

// --- Optimized Streaming Renderer ---
class StreamingRenderer {
    constructor(container) {
        this.container = container;
        this.fullText = '';
        this.tokenCache = [];
        this.blockElements = [];
        this.tokenQueue = [];
        this.rafId = null;
        this.isComplete = false;
        this._lastRenderedText = '';
        this.cursor = document.createElement('span');
        this.cursor.className = 'streaming-cursor';
        this.cursor.textContent = '▌';
    }

    push(chunk) {
        if (!chunk) return;
        this.tokenQueue.push(chunk);
        if (!this.rafId) {
            this.rafId = requestAnimationFrame(() => this.processQueue());
        }
    }

    processQueue() {
        if (this.tokenQueue.length === 0) {
            this.rafId = null;
            return;
        }

        const tokensToProcess = Math.max(5, Math.ceil(this.tokenQueue.length / 4));
        const chunk = this.tokenQueue.splice(0, tokensToProcess).join('');
        this.fullText += chunk;
        this.render();

        if (this.tokenQueue.length > 0) {
            this.rafId = requestAnimationFrame(() => this.processQueue());
        } else {
            this.rafId = null;
        }
    }

    render(isDone = false) {
        if (!isDone && this.fullText === this._lastRenderedText) return;
        this._lastRenderedText = this.fullText;
        this.isComplete = isDone;
        let textToParse = this.fullText;

        if (!isDone) {
            const matches = textToParse.match(/(?:^|\n)```/g) || [];
            if (matches.length % 2 !== 0 && !textToParse.endsWith('```')) {
                textToParse += '\n```';
            }
        }

        const tokens = marked.lexer(textToParse);
        const maxLen = Math.max(tokens.length, this.blockElements.length);

        for (let i = 0; i < maxLen; i++) {
            const isLast = i === tokens.length - 1;
            const token = tokens[i];
            const cachedToken = this.tokenCache[i];

            if (token && cachedToken && cachedToken.raw === token.raw) {
                if (!isLast) continue;
                if (this.cursor.parentNode) {
                    const target = this.blockElements[i].lastElementChild || this.blockElements[i];
                    if (!target.contains(this.cursor)) {
                        target.appendChild(this.cursor);
                    }
                }
                continue;
            }

            if (token) {
                let element = this.blockElements[i];
                if (!element) {
                    element = document.createElement('div');
                    element.className = 'markdown-block';
                    this.container.appendChild(element);
                    this.blockElements[i] = element;
                }
                safeRender(element, marked.parser([token]));
                this.tokenCache[i] = { raw: token.raw };

                if (isLast && !isDone) {
                    const target = element.lastElementChild || element;
                    target.appendChild(this.cursor);
                }
            } else {
                if (this.blockElements[i]) {
                    this.blockElements[i].remove();
                    this.blockElements[i] = null;
                    this.tokenCache[i] = null;
                }
            }
        }

        while (this.blockElements.length > tokens.length) {
            const el = this.blockElements.pop();
            if (el) el.remove();
            this.tokenCache.pop();
        }

        if (this.tokenCache.length > 200) {
            this.tokenCache = this.tokenCache.slice(-100);
            this.blockElements = this.blockElements.slice(-100);
        }

        this.scrollToBottom();
    }

    scrollToBottom() {
        const chatArea = document.getElementById('chatMessages');
        if (chatArea) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    }

    finish(finalText = null) {
        if (this.isComplete) return;
        if (finalText !== null) this.fullText = finalText;
        this.render(true);
        if (this.cursor.parentNode) {
            this.cursor.parentNode.removeChild(this.cursor);
        }
    }
}

async function sendMessage() {
    const text = userInput.value;
    if (text.trim() === '') return;

    const { messageDiv: userDiv, contentDiv: userContent } = createMessageElement('user');
    userContent.textContent = text;
    chatMessages.appendChild(userDiv);

    userInput.value = '';
    userInput.style.height = 'auto';
    sendButton.disabled = true;

    const { messageDiv: aiDiv, contentDiv: aiContent } = createMessageElement('ai');
    showTypingIndicator(aiContent);
    chatMessages.appendChild(aiDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });

    try {
        let fullResponse = '';
        let continuationCount = 0;
        const MAX_CONTINUATIONS = 3;
        const renderer = new StreamingRenderer(aiContent);
        aiContent.innerHTML = '';

        let isContinuation = false;
        let streamDone = false;
        let sseBuffer = '';
        let eventBuffer = [];

        while (!streamDone && continuationCount < MAX_CONTINUATIONS) {
            const response = await fetch(`${BACKEND_URL}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    session_id: sessionId,
                    is_continuation: isContinuation
                })
            });

            if (!response.ok) throw new Error("Backend unavailable");

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                sseBuffer += decoder.decode(value, { stream: true });

                const events = sseBuffer.split('\n\n');
                sseBuffer = events.pop() || '';

                for (const event of events) {
                    if (!event.trim()) continue;

                    const lines = event.split('\n');
                    let dataStr = '';
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            dataStr += line.substring(6).trim();
                        }
                    }

                    if (dataStr === '[DONE]') {
                        streamDone = true;
                        break;
                    }

                    if (!dataStr) continue;

                    try {
                        const parsed = JSON.parse(dataStr);
                        if (parsed.content) {
                            renderer.push(parsed.content);
                            fullResponse += parsed.content;
                        }
                        if (parsed.refined && parsed.full) {
                            fullResponse = parsed.full;
                            renderer.finish(fullResponse);
                        }
                        if (parsed.done) {
                            streamDone = true;
                        }
                    } catch (e) {
                        eventBuffer.push(event);
                    }
                }

                // Retry buffered events
                const remaining = [];
                for (const ev of eventBuffer) {
                    const lines = ev.split('\n');
                    let dataStr = '';
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            dataStr += line.substring(6).trim();
                        }
                    }
                    if (dataStr) {
                        try {
                            const parsed = JSON.parse(dataStr);
                            if (parsed.content) {
                                renderer.push(parsed.content);
                                fullResponse += parsed.content;
                            }
                            if (parsed.done) streamDone = true;
                        } catch (e) {
                            remaining.push(ev);
                        }
                    }
                }
                eventBuffer = remaining;

                if (streamDone) break;
            }

            if (!streamDone) {
                isContinuation = true;
                continuationCount++;
                await new Promise(r => setTimeout(r, 500));
            }
        }

        renderer.finish();

    } catch (error) {
        console.error("Chat error:", error);
        safeRender(aiContent, marked.parse('**Error:** Could not reach the backend server.'));
    }
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendButton.disabled) sendMessage();
    }
});

uploadButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const { messageDiv, contentDiv } = createMessageElement('user');
    contentDiv.textContent = `Uploading: ${file.name}...`;
    chatMessages.appendChild(messageDiv);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    try {
        const response = await fetch(`${BACKEND_URL}/upload`, { method: 'POST', body: formData });
        if (response.ok) {
            const { messageDiv: aiDiv, contentDiv: aiContent } = createMessageElement('ai');
            aiContent.textContent = `Successfully processed ${file.name}.`;
            chatMessages.appendChild(aiDiv);
        }
    } catch (err) {
        console.error(err);
    }
    fileInput.value = '';
});

userInput.focus();
