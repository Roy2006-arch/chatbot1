// --- Configure marked.js with Custom Renderer and Highlight.js ---
marked.use({
    renderer: {
        code(code, language) {
            const validLang = language && hljs.getLanguage(language) ? language : 'plaintext';
            const highlighted = validLang === 'plaintext' ? hljs.highlightAuto(code).value : hljs.highlight(code, { language: validLang }).value;
            
            return `
<div class="code-block-wrapper">
    <div class="code-block-header">
        <span class="code-lang">${validLang}</span>
        <button class="copy-btn" onclick="copyCode(this)">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            Copy
        </button>
    </div>
    <pre><code class="hljs language-${validLang}">${highlighted}</code></pre>
</div>`;
        }
    }
});

marked.setOptions({
    breaks: true,        // Convert \n to <br>
    gfm: true,           // GitHub Flavoured Markdown (tables, code blocks, etc.)
    sanitize: false      // We trust our own backend output
});

// Global copy function
window.copyCode = function(button) {
    const wrapper = button.closest('.code-block-wrapper');
    const codeEl = wrapper.querySelector('code');
    const text = codeEl.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        const originalHTML = button.innerHTML;
        button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
        setTimeout(() => button.innerHTML = originalHTML, 2000);
    });
};

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const themeToggle = document.getElementById('themeToggle');
const sunIcon = document.getElementById('sunIcon');
const moonIcon = document.getElementById('moonIcon');
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');

// --- Configure Backend URL ---
// If running locally, use localhost. If on Netlify, use the Hugging Face Space URL.
const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://kaustav2006-chatbot-api.hf.space';

const sessionId = Math.random().toString(36).substring(7);

// --- Theme Toggle Logic ---
// Check local storage or system preference
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
    if (theme === 'dark') {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    } else {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    }
}

// --- Textarea Auto-resize & Button State ---
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendButton.disabled = this.value.trim() === '';
});

// --- Chat Logic ---
function createMessageElement(sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'ai-message');
    
    // Create Avatar
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
    contentDiv.innerHTML = `
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
}

// --- Code Detection Logic ---
function detectCodeAndLanguage(text) {
    const lines = text.split('\n');
    
    // Fast fail for short text
    if (lines.length === 1 && text.length < 20) return null;

    const patterns = {
        javascript: /^(const|let|var|function|import|export|console\.log)\b/m,
        python: /^(def|class|import|from|print|if __name__|elif|yield)\b/m,
        java: /^(public|private|protected|class|interface|static|void|System\.out)\b/m,
        cpp: /^(#include|using namespace|int main|std::|cout)\b/m,
        html: /<\/?(html|body|div|span|p|a|script|style)[^>]*>/m,
        css: /([.#\w][^{]+)\{([^}]+)\}/m,
        sql: /^(SELECT|UPDATE|DELETE|INSERT|CREATE|DROP|ALTER)\b/i
    };

    // Calculate structural indicators
    const indentedLines = lines.filter(line => /^( {2,}|\t)/.test(line));
    const indentationRatio = lines.length > 0 ? indentedLines.length / lines.length : 0;
    
    const symbols = text.match(/[{}[\]();=+\-*/&|!<>]/g);
    const symbolDensity = symbols ? symbols.length / text.length : 0;

    // Check regex matches
    for (const [lang, regex] of Object.entries(patterns)) {
        if (regex.test(text)) return lang;
    }

    // Heuristic fallbacks
    if (indentationRatio > 0.4 || symbolDensity > 0.1) return 'code';
    
    return null;
}

function renderUserMessage(text, container) {
    container.innerHTML = '';
    
    // 1. Check if user manually provided markdown code blocks
    if (text.includes('```')) {
        const p = document.createElement('div');
        p.textContent = text; // Safe text rendering
        container.appendChild(p);
        return;
    }

    // 2. Auto-detect code blocks based on empty lines (double newline)
    const blocks = text.split(/\n{2,}/);
    
    blocks.forEach(block => {
        const lang = detectCodeAndLanguage(block);
        
        if (lang) {
            const wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';
            
            const header = document.createElement('div');
            header.className = 'code-block-header';
            header.innerHTML = `
                <span class="code-lang">${lang}</span>
                <button class="copy-btn" onclick="copyCode(this)">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    Copy
                </button>
            `;

            const pre = document.createElement('pre');
            const code = document.createElement('code');
            code.className = `hljs language-${lang}`;
            code.textContent = block; // Safely insert as text (prevents XSS)
            
            // Apply Highlight.js to the user's code block
            hljs.highlightElement(code);
            
            pre.appendChild(code);
            wrapper.appendChild(header);
            wrapper.appendChild(pre);
            container.appendChild(wrapper);
        } else {
            const div = document.createElement('div');
            div.textContent = block; // Safely insert as text
            container.appendChild(div);
            
            // Add spacing between blocks
            const br = document.createElement('br');
            container.appendChild(br);
            const br2 = document.createElement('br');
            container.appendChild(br2);
        }
    });
    
    // Remove trailing breaks
    while (container.lastChild && container.lastChild.tagName === 'BR') {
        container.removeChild(container.lastChild);
    }
}

async function sendMessage() {
    const text = userInput.value;
    if (text.trim() === '') return;
    
    // Add user message
    const { messageDiv: userDiv, contentDiv: userContent } = createMessageElement('user');
    renderUserMessage(text, userContent);
    chatMessages.appendChild(userDiv);
    
    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendButton.disabled = true;
    
    // Scroll
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    
    // Add AI message with typing indicator
    const { messageDiv: aiDiv, contentDiv: aiContent } = createMessageElement('ai');
    showTypingIndicator(aiContent);
    chatMessages.appendChild(aiDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    
    try {
        const response = await fetch(`${BACKEND_URL}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });
        
        // Remove typing indicator once stream starts
        aiContent.innerHTML = '';

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;
        let sseBuffer = '';      // Buffer for incomplete SSE lines
        let fullResponse = '';   // Accumulate full text for final Markdown render
        let tokenCount = 0;      // Track approximate token count for truncation detection

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                sseBuffer += decoder.decode(value, { stream: !done });
                const lines = sseBuffer.split('\n');

                // Keep the last incomplete line in the buffer
                sseBuffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '').trim();
                        if (dataStr === '[DONE]') {
                            done = true;
                            break;
                        }
                        try {
                            const parsed = JSON.parse(dataStr);
                            if (parsed.content) {
                                fullResponse += parsed.content;
                                tokenCount++;
                            }
                            
                            // If the backend sends a refined (final corrected) version, use that instead
                            if (parsed.refined && parsed.full) {
                                fullResponse = parsed.full;
                            }

                            // Re-render the full accumulated Markdown on every token
                            // This gives a live-streaming Markdown effect
                            aiContent.innerHTML = marked.parse(fullResponse);
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } catch (e) {
                            console.error("Parse error:", e, "on string:", dataStr);
                        }
                    }
                }
            }
        }

        // --- FIX 1: Hallucination guard fired (empty response) ---
        if (fullResponse.trim() === '') {
            aiContent.innerHTML = marked.parse(
                '⚠️ *My response was filtered for safety. Please try rephrasing your message.*'
            );
        }

        // --- FIX 2: Truncation detection (response likely hit max_new_tokens) ---
        // If the response doesn't end with sentence-ending punctuation, it was cut off
        const trimmed = fullResponse.trimEnd();
        const endsCleanly = /[.!?`\])]$/.test(trimmed);
        if (trimmed.length > 0 && !endsCleanly && tokenCount >= 140) {
            aiContent.innerHTML += marked.parse(
                '\n\n---\n*✂️ Response was cut short due to length. Ask me to **"continue"** for more.*'
            );
        }

    } catch (error) {
        aiContent.innerHTML = marked.parse('❌ **Error:** Could not reach the backend server. Please ensure it is running on `127.0.0.1:8000`.');
    }
}

sendButton.addEventListener('click', sendMessage);

// Handle Enter to send, Shift+Enter for new line, Tab for indentation
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent default new line
        if (!sendButton.disabled) {
            sendMessage();
        }
    } else if (e.key === 'Tab') {
        e.preventDefault();
        const start = e.target.selectionStart;
        const end = e.target.selectionEnd;
        e.target.value = e.target.value.substring(0, start) + "\t" + e.target.value.substring(end);
        e.target.selectionStart = e.target.selectionEnd = start + 1;
        // Trigger input event to update button state and textarea height
        e.target.dispatchEvent(new Event('input'));
    }
});

// --- Document Upload Logic ---
uploadButton.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // UI Feedback
    const { messageDiv, contentDiv } = createMessageElement('user');
    contentDiv.innerHTML = `<em>Uploading document: ${file.name}...</em>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    
    try {
        const response = await fetch(`${BACKEND_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if(response.ok) {
            const { messageDiv: aiDiv, contentDiv: aiContent } = createMessageElement('ai');
            aiContent.textContent = `Successfully processed ${file.name}. You can now ask me questions about it!`;
            chatMessages.appendChild(aiDiv);
        } else {
            throw new Error("Upload failed");
        }
    } catch (err) {
        console.error(err);
        const { messageDiv: aiDiv, contentDiv: aiContent } = createMessageElement('ai');
        aiContent.textContent = `Sorry, I failed to upload ${file.name}. Check if the backend is running.`;
        chatMessages.appendChild(aiDiv);
    }
    
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    fileInput.value = ''; // Reset input
});

userInput.focus();
