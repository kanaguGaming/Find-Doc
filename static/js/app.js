/* ============================================================
   MedVision AI — Frontend Application Logic
   ============================================================ */

// ----------------------------------------------------------------
// STATE
// ----------------------------------------------------------------
const state = {
    currentCondition: null,
    currentModality:  null,
    bodyRegion:       null,
    isAnalyzing:      false,
    isChatLoading:    false,
};

// ----------------------------------------------------------------
// DOM REFS
// ----------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const els = {
    uploadForm:      $('uploadForm'),
    imageInput:      $('imageInput'),
    uploadLabel:     $('uploadLabel'),
    uploadTitle:     null,  // set after DOM ready
    analyzeBtn:      $('analyzeBtn'),
    analysisLoader:  $('analysisLoader'),
    resultCard:      $('resultCard'),

    // Loader steps
    lstep1:          $('lstep1'),
    lstep2:          $('lstep2'),
    lstep3:          $('lstep3'),

    // Result fields
    badgeModality:   $('badgeModality'),
    badgeRegion:     $('badgeRegion'),
    badgeSeverity:   $('badgeSeverity'),
    previewImg:      $('previewImg'),
    heatmapBox:      $('heatmapBox'),
    lblPrediction:   $('lblPrediction'),
    confidenceFill:  $('confidenceFill'),
    lblConfidence:   $('lblConfidence'),
    findingsList:    $('findingsList'),

    // Chat
    chatMessages:    $('chatMessages'),
    chatInput:       $('chatInput'),
    sendChatBtn:     $('sendChatBtn'),
    typingIndicator: $('typingIndicator'),

    // Header status
    statusDot:       $('statusDot'),
    statusLabel:     $('statusLabel'),
};

// ----------------------------------------------------------------
// STATUS HELPERS
// ----------------------------------------------------------------
function setStatus(type, text) {
    els.statusDot.className   = 'status-dot' + (type ? ` ${type}` : '');
    els.statusLabel.textContent = text;
}

// ----------------------------------------------------------------
// LOADER STEP SEQUENCER
// ----------------------------------------------------------------
function runLoaderSequence() {
    // Reset
    [els.lstep1, els.lstep2, els.lstep3].forEach(el => {
        el.classList.remove('active', 'done');
    });

    els.lstep1.classList.add('active');

    const t1 = setTimeout(() => {
        els.lstep1.classList.remove('active');
        els.lstep1.classList.add('done');
        els.lstep2.classList.add('active');
    }, 2500);

    const t2 = setTimeout(() => {
        els.lstep2.classList.remove('active');
        els.lstep2.classList.add('done');
        els.lstep3.classList.add('active');
    }, 5000);

    return () => { clearTimeout(t1); clearTimeout(t2); };
}

// ----------------------------------------------------------------
// FILE INPUT HANDLER
// ----------------------------------------------------------------
els.imageInput.addEventListener('change', () => {
    const file = els.imageInput.files[0];
    if (!file) return;

    // Update label
    const titleEl = els.uploadLabel.querySelector('.upload-title');
    if (titleEl) titleEl.textContent = `✓ ${file.name}`;
    els.uploadLabel.classList.add('has-file');
    els.analyzeBtn.disabled = false;
});

// Drag-and-drop on upload label
els.uploadLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.uploadLabel.style.borderColor = 'var(--accent-1)';
});
els.uploadLabel.addEventListener('dragleave', () => {
    els.uploadLabel.style.borderColor = '';
});
els.uploadLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    els.uploadLabel.style.borderColor = '';
    const dt = e.dataTransfer;
    if (dt.files && dt.files.length) {
        els.imageInput.files = dt.files;
        els.imageInput.dispatchEvent(new Event('change'));
    }
});

// ----------------------------------------------------------------
// IMAGE ANALYSIS SUBMISSION
// ----------------------------------------------------------------
els.uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!els.imageInput.files.length || state.isAnalyzing) return;

    state.isAnalyzing = true;
    setStatus('analyzing', 'Analyzing…');

    // Show loader, hide previous results
    els.resultCard.classList.add('hidden');
    els.analysisLoader.classList.remove('hidden');
    els.analyzeBtn.disabled = true;

    const cancelLoader = runLoaderSequence();
    const formData = new FormData();
    formData.append('file', els.imageInput.files[0]);

    // Render image preview immediately
    const reader = new FileReader();
    reader.onload = (ev) => { els.previewImg.src = ev.target.result; };
    reader.readAsDataURL(els.imageInput.files[0]);

    try {
        const resp = await fetch('/api/triage/analyze', {
            method: 'POST',
            body:   formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        cancelLoader();

        // ---- Persist state for chat ----
        state.currentCondition = data.prediction;
        state.currentModality  = data.modality;
        state.bodyRegion       = data.body_region;

        // ---- Populate result card ----
        renderAnalysisResults(data);

        // ---- Enable chat ----
        els.chatInput.disabled    = false;
        els.sendChatBtn.disabled  = false;
        els.chatInput.placeholder = 'Ask about the findings…';
        els.chatInput.focus();

        // ---- Inject triage note into chat ----
        appendBotMessage(data.initial_triage_note);

        setStatus('', 'Analysis Complete');

    } catch (err) {
        cancelLoader();
        console.error('Analysis error:', err);
        appendBotMessage(`⚠️ Analysis failed: ${err.message}. Please check your API key and try again.`);
        setStatus('error', 'Analysis Failed');
    } finally {
        state.isAnalyzing = false;
        els.analysisLoader.classList.add('hidden');
        els.analyzeBtn.disabled = false;
    }
});

// ----------------------------------------------------------------
// RENDER ANALYSIS RESULTS
// ----------------------------------------------------------------
function renderAnalysisResults(data) {
    // Badges
    els.badgeModality.textContent = data.modality    || '—';
    els.badgeRegion.textContent   = data.body_region || '—';

    const sev = data.severity || 'Unknown';
    els.badgeSeverity.textContent        = sev;
    els.badgeSeverity.setAttribute('data-sev', sev);

    // Condition
    els.lblPrediction.textContent = data.prediction || 'Not determined';

    // Confidence bar (animate)
    const conf = parseFloat(data.confidence) || 0;
    els.lblConfidence.textContent = `${conf.toFixed(1)}%`;
    // Trigger animation after a short delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            els.confidenceFill.style.width = `${Math.min(conf, 100)}%`;
        }, 100);
    });

    // Findings list
    els.findingsList.innerHTML = '';
    const findings = Array.isArray(data.findings) ? data.findings : ['No observations extracted.'];
    findings.forEach((f, i) => {
        const li = document.createElement('li');
        li.textContent = f;
        li.style.animationDelay = `${i * 80}ms`;
        els.findingsList.appendChild(li);
    });

    // Heatmap overlay
    if (data.heatmap) {
        const h = data.heatmap;
        els.heatmapBox.style.top    = `${h.top}%`;
        els.heatmapBox.style.left   = `${h.left}%`;
        els.heatmapBox.style.width  = `${h.width}%`;
        els.heatmapBox.style.height = `${h.height}%`;
    }

    // Show card
    els.resultCard.classList.remove('hidden');
}

// ----------------------------------------------------------------
// CHAT — Send Message
// ----------------------------------------------------------------
async function sendChatMessage() {
    const query = els.chatInput.value.trim();
    if (!query || state.isChatLoading) return;

    appendUserMessage(query);
    els.chatInput.value = '';
    state.isChatLoading = true;

    // Show typing indicator
    els.typingIndicator.classList.remove('hidden');
    scrollChat();

    try {
        const resp = await fetch('/api/triage/chat', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({
                message:            query,
                detected_condition: state.currentCondition,
                detected_modality:  state.currentModality,
                body_region:        state.bodyRegion,
            }),
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const data = await resp.json();
        els.typingIndicator.classList.add('hidden');
        appendBotMessage(data.response);

        // Show doctor recommendations only when needed
        if (data.actions_required && data.suggested_doctors && data.suggested_doctors.length > 0) {
            appendDoctorCards(data.suggested_doctors);
        }

    } catch (err) {
        els.typingIndicator.classList.add('hidden');
        console.error('Chat error:', err);
        appendBotMessage('⚠️ Unable to reach the triage server. Please try again.');
    } finally {
        state.isChatLoading = false;
    }
}

// Send on button click
els.sendChatBtn.addEventListener('click', sendChatMessage);

// Send on Enter key (Shift+Enter = newline behavior skipped since it's an input)
els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

// ----------------------------------------------------------------
// CHAT MESSAGE RENDERERS
// ----------------------------------------------------------------
function appendBotMessage(text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg msg-bot';
    wrapper.innerHTML = `
        <div class="msg-avatar bot-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
        </div>
        <div class="msg-bubble">${escapeHtml(text)}</div>
    `;
    els.chatMessages.appendChild(wrapper);
    scrollChat();
}

function appendUserMessage(text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg msg-user';
    wrapper.innerHTML = `
        <div class="msg-avatar user-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
        </div>
        <div class="msg-bubble">${escapeHtml(text)}</div>
    `;
    els.chatMessages.appendChild(wrapper);
    scrollChat();
}

function appendDoctorCards(doctors) {
    const group = document.createElement('div');
    group.className = 'doctor-card-group msg';
    group.style.flexDirection = 'column';

    const titleEl = document.createElement('div');
    titleEl.className = 'doctor-card-title';
    titleEl.textContent = '📍 Recommended Specialists Near You';
    group.appendChild(titleEl);

    doctors.forEach(doc => {
        const card = document.createElement('div');
        card.className = 'doctor-card';
        card.innerHTML = `
            <div class="doctor-name">${escapeHtml(doc.name)}</div>
            <div class="doctor-spec">${escapeHtml(doc.specialty || '')}</div>
            <div class="doctor-meta">${escapeHtml(doc.address)} &bull; ${escapeHtml(doc.distance)}</div>
        `;
        group.appendChild(card);
    });

    els.chatMessages.appendChild(group);
    scrollChat();
}

// ----------------------------------------------------------------
// UTILITIES
// ----------------------------------------------------------------
function scrollChat() {
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str
        .replace(/&/g,  '&amp;')
        .replace(/</g,  '&lt;')
        .replace(/>/g,  '&gt;')
        .replace(/"/g,  '&quot;')
        .replace(/'/g,  '&#x27;');
}