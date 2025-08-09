(() => {
  const dom = {
    sessionInfo: document.getElementById('sessionInfo'),
    sessionIdInput: document.getElementById('sessionIdInput'),
    generateSessionId: document.getElementById('generateSessionId'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    uploadStatus: document.getElementById('uploadStatus'),
    step2: document.getElementById('step2'),
    chunkSize: document.getElementById('chunkSize'),
    chunkOverlap: document.getElementById('chunkOverlap'),
    loadBtn: document.getElementById('loadBtn'),
    loadSpinner: document.getElementById('loadSpinner'),
    loadStatus: document.getElementById('loadStatus'),
    step3: document.getElementById('step3'),
    questionInput: document.getElementById('questionInput'),
    askBtn: document.getElementById('askBtn'),
    askStatus: document.getElementById('askStatus'),
    answer: document.getElementById('answer'),
    sources: document.getElementById('sources'),
    resetBtn: document.getElementById('resetBtn'),
  };

  const state = {
    apiBaseUrl: window.location.origin,
    sessionId: window.localStorage.getItem('sessionId') || '',
  };

  function setHidden(element, value) {
    element.classList.toggle('hidden', value);
  }

  function updateSessionInfo() {
    if (!state.sessionId) {
      dom.sessionInfo.textContent = '';
      setHidden(dom.sessionInfo, true);
      return;
    }
    dom.sessionInfo.innerHTML = `Using session <strong>${escapeHtml(state.sessionId)}</strong>`;
    setHidden(dom.sessionInfo, false);
  }

  function escapeHtml(unsafe) {
    return String(unsafe)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function saveSessionId(sessionId) {
    state.sessionId = sessionId || '';
    dom.sessionIdInput.value = state.sessionId;
    if (state.sessionId) {
      window.localStorage.setItem('sessionId', state.sessionId);
    } else {
      window.localStorage.removeItem('sessionId');
    }
    updateSessionInfo();
  }

  function init() {
    dom.sessionIdInput.value = state.sessionId;
    updateSessionInfo();

    if (state.sessionId) {
      setHidden(dom.step2, false);
      setHidden(dom.step3, false);
    }

    dom.generateSessionId.addEventListener('click', () => {
      // Prefer crypto.randomUUID where available
      const newId = (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : `sess_${Date.now()}`;
      dom.sessionIdInput.value = newId;
    });

    dom.fileInput.addEventListener('change', () => {
      dom.uploadBtn.disabled = dom.fileInput.files.length === 0;
      dom.uploadStatus.textContent = '';
    });

    dom.uploadBtn.addEventListener('click', onUpload);
    dom.loadBtn.addEventListener('click', onLoad);
    dom.askBtn.addEventListener('click', onAsk);
    dom.questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        onAsk();
      }
    });
    dom.resetBtn.addEventListener('click', resetSession);
  }

  async function onUpload() {
    const files = Array.from(dom.fileInput.files || []);
    if (files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
      formData.append('files[]', file, file.name);
    }
    const providedSessionId = dom.sessionIdInput.value.trim();
    if (providedSessionId) {
      formData.append('session_id', providedSessionId);
    }

    dom.uploadBtn.disabled = true;
    dom.uploadStatus.textContent = 'Uploading…';

    try {
      const res = await fetch(joinUrl(state.apiBaseUrl, '/upload'), {
        method: 'POST',
        body: formData,
      });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(messageFrom(data, res.status));

      const newSessionId = data.session_id || providedSessionId;
      if (!newSessionId) {
        throw new Error('Upload succeeded but no session_id was returned.');
      }
      saveSessionId(newSessionId);

      dom.uploadStatus.textContent = `Uploaded ${files.length} file(s). Session: ${newSessionId}`;
      setHidden(dom.step2, false);
    } catch (err) {
      dom.uploadStatus.textContent = `Upload failed: ${err.message}`;
    } finally {
      dom.uploadBtn.disabled = false;
    }
  }

  async function onLoad() {
    if (!state.sessionId) {
      dom.loadStatus.textContent = 'No session_id. Upload first.';
      return;
    }
    const payload = {
      session_id: state.sessionId,
      chunk_size: toNumber(dom.chunkSize.value, 500),
      chunk_overlap: toNumber(dom.chunkOverlap.value, 50),
    };

    dom.loadBtn.disabled = true;
    setHidden(dom.loadSpinner, false);
    dom.loadStatus.textContent = 'Indexing…';

    try {
      const res = await fetch(joinUrl(state.apiBaseUrl, '/load'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(messageFrom(data, res.status));

      dom.loadStatus.textContent = 'Indexed successfully.';
      setHidden(dom.step3, false);
    } catch (err) {
      dom.loadStatus.textContent = `Load failed: ${err.message}`;
    } finally {
      dom.loadBtn.disabled = false;
      setHidden(dom.loadSpinner, true);
    }
  }

  async function onAsk() {
    const question = dom.questionInput.value.trim();
    if (!question) return;
    if (!state.sessionId) {
      dom.askStatus.textContent = 'No session_id. Upload and load first.';
      return;
    }

    const payload = {
      session_id: state.sessionId,
      question,
      top_k: 4,
      include_sources: true,
    };

    dom.askBtn.disabled = true;
    dom.askStatus.textContent = 'Thinking…';
    dom.answer.textContent = '';
    dom.sources.innerHTML = '';

    try {
      const res = await fetch(joinUrl(state.apiBaseUrl, '/ask'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(messageFrom(data, res.status));

      const answerText = pickFirst(data, ['answer', 'text', 'result', 'output']) || '';
      dom.answer.textContent = answerText;

      const sources = pickSources(data);
      renderSources(sources);

      dom.askStatus.textContent = '';
    } catch (err) {
      dom.askStatus.textContent = `Ask failed: ${err.message}`;
    } finally {
      dom.askBtn.disabled = false;
    }
  }

  function pickSources(data) {
    // Attempt to normalize common shapes
    if (Array.isArray(data.sources)) return data.sources;
    if (Array.isArray(data.source_documents)) return data.source_documents;
    if (Array.isArray(data.documents)) return data.documents;
    if (data.context && Array.isArray(data.context.sources)) return data.context.sources;
    return [];
  }

  function renderSources(sources) {
    dom.sources.innerHTML = '';
    if (!sources || sources.length === 0) return;

    for (const src of sources) {
      const div = document.createElement('div');
      div.className = 'source-item';
      const title = src.title || src.metadata?.title || src.name || 'Source';
      const sourcePath = src.source || src.metadata?.source || src.path || src.id || '';
      const page = src.page || src.metadata?.page || src.metadata?.loc?.page || undefined;
      const snippet = src.snippet || src.text || src.content || '';

      div.innerHTML = `
        <h4>${escapeHtml(String(title))}</h4>
        <div class="kv">${escapeHtml(String(sourcePath))}${page !== undefined ? ` (p. ${escapeHtml(String(page))})` : ''}</div>
        ${snippet ? `<div style="margin-top:6px">${escapeHtml(String(snippet))}</div>` : ''}
      `;
      dom.sources.appendChild(div);
    }
  }

  function resetSession() {
    saveSessionId('');
    dom.answer.textContent = '';
    dom.sources.innerHTML = '';
    dom.uploadStatus.textContent = '';
    dom.loadStatus.textContent = '';
    dom.askStatus.textContent = '';
    setHidden(dom.step2, true);
    setHidden(dom.step3, true);
  }

  function toNumber(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function joinUrl(base, path) {
    try {
      return new URL(path, base).toString();
    } catch {
      return `${base.replace(/\/$/, '')}${path.startsWith('/') ? '' : '/'}${path}`;
    }
  }

  async function safeJson(res) {
    const text = await res.text();
    try { return text ? JSON.parse(text) : {}; }
    catch { return { _raw: text }; }
  }

  function messageFrom(data, status) {
    if (data && typeof data.error === 'string') return data.error;
    if (data && typeof data.message === 'string') return data.message;
    return `HTTP ${status}`;
  }

  function pickFirst(obj, keys) {
    for (const k of keys) {
      if (obj && typeof obj[k] === 'string') return obj[k];
    }
    return '';
  }

  // Init
  init();
})();

