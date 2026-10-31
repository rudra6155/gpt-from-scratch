// gpt-scratch — Model Inspector & Inference Workbench Client

document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  const views = document.querySelectorAll('.workbench-view');

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      navTabs.forEach(t => t.classList.remove('active'));
      views.forEach(v => v.classList.add('hidden'));
      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetView = document.getElementById(targetId);
      if (targetView) {
        targetView.classList.remove('hidden');
      }
    });
  });

  // Controls & Inputs
  const promptInput = document.getElementById('prompt-input');
  const promptCharCount = document.getElementById('prompt-char-count');
  const presetChips = document.querySelectorAll('.preset-chip');

  const tempSlider = document.getElementById('temp-slider');
  const tempNum = document.getElementById('temp-num');
  const topkSlider = document.getElementById('topk-slider');
  const topkNum = document.getElementById('topk-num');
  const toppSlider = document.getElementById('topp-slider');
  const toppNum = document.getElementById('topp-num');
  const maxtokSlider = document.getElementById('maxtok-slider');
  const maxtokNum = document.getElementById('maxtok-num');
  const seedInput = document.getElementById('seed-input');
  const btnSeedRandom = document.getElementById('btn-seed-random');

  const btnGenerate = document.getElementById('btn-generate');
  const btnStop = document.getElementById('btn-stop');
  const btnClear = document.getElementById('btn-clear');
  const btnCopy = document.getElementById('btn-copy');

  // Stages
  const btnViewText = document.getElementById('btn-view-text');
  const btnViewTokens = document.getElementById('btn-view-tokens');
  const stageText = document.getElementById('stage-text');
  const stageTokens = document.getElementById('stage-tokens');
  const outputText = document.getElementById('output-text');
  const tokenGrid = document.getElementById('token-grid');
  const tokenDetailContent = document.getElementById('token-detail-content');

  // Telemetry
  const deviceVal = document.getElementById('device-val');
  const paramsVal = document.getElementById('params-val');
  const statTokens = document.getElementById('stat-tokens');
  const statSpeed = document.getElementById('stat-speed');
  const statLatency = document.getElementById('stat-latency');

  let activeAbort = null;
  let accumulatedText = '';
  let tokenRecords = [];

  // Sync Slider <-> Number Inputs
  function bindSync(slider, num, isFloat = true) {
    slider.addEventListener('input', () => {
      num.value = isFloat ? parseFloat(slider.value).toFixed(2) : slider.value;
    });
    num.addEventListener('input', () => {
      slider.value = num.value;
    });
  }
  bindSync(tempSlider, tempNum, true);
  bindSync(topkSlider, topkNum, false);
  bindSync(toppSlider, toppNum, true);
  bindSync(maxtokSlider, maxtokNum, false);

  btnSeedRandom.addEventListener('click', () => {
    seedInput.value = '';
  });

  // Prompt Char Count
  function updateCharCount() {
    promptCharCount.textContent = `${promptInput.value.length} chars`;
  }
  promptInput.addEventListener('input', updateCharCount);
  updateCharCount();

  // Presets
  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      promptInput.value = chip.getAttribute('data-prompt');
      updateCharCount();
      promptInput.focus();
    });
  });

  // View Switching: Text vs Tokens
  btnViewText.addEventListener('click', () => {
    btnViewText.classList.add('active');
    btnViewTokens.classList.remove('active');
    stageText.classList.remove('hidden');
    stageTokens.classList.add('hidden');
  });

  btnViewTokens.addEventListener('click', () => {
    btnViewTokens.classList.add('active');
    btnViewText.classList.remove('active');
    stageText.classList.add('hidden');
    stageTokens.classList.remove('hidden');
    renderTokens();
  });

  // Render Token Grid
  function renderTokens() {
    tokenGrid.innerHTML = '';
    if (!tokenRecords.length) {
      tokenGrid.innerHTML = '<div class="token-placeholder">No generated tokens to display. Run inference first.</div>';
      tokenDetailContent.textContent = 'Hover or click a token above to inspect metadata.';
      return;
    }

    tokenRecords.forEach((tok, idx) => {
      const chip = document.createElement('div');
      chip.className = 'token-chip';
      chip.dataset.index = idx;

      const pieceSpan = document.createElement('span');
      // Format whitespace for visual clarity
      const repr = tok.token.replace(/\n/g, '↵\n').replace(/ /g, '·');
      pieceSpan.textContent = repr;

      const idSpan = document.createElement('span');
      idSpan.className = 'token-chip-id';
      idSpan.textContent = `#${tok.id}`;

      chip.appendChild(pieceSpan);
      chip.appendChild(idSpan);

      chip.addEventListener('click', () => {
        document.querySelectorAll('.token-chip').forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');
        tokenDetailContent.innerHTML = `<strong>Token #${idx + 1}</strong> | ID: <code>${tok.id}</code> | Text Piece: <code>${JSON.stringify(tok.token)}</code> | Byte Length: ${new TextEncoder().encode(tok.token).length} bytes`;
      });

      tokenGrid.appendChild(chip);
    });
  }

  // Clear Output
  btnClear.addEventListener('click', () => {
    accumulatedText = '';
    tokenRecords = [];
    outputText.textContent = 'Select a prompt or click Generate Continuation to evaluate autoregressive sampling.';
    tokenGrid.innerHTML = '<div class="token-placeholder">No generated tokens to display. Run inference first.</div>';
    tokenDetailContent.textContent = 'Hover or click a token above to inspect metadata.';
    statTokens.textContent = '0 tokens';
    statSpeed.textContent = '0.0 tok/s';
    statLatency.textContent = '0.00s';
  });

  // Copy Output
  btnCopy.addEventListener('click', async () => {
    if (!accumulatedText) return;
    try {
      await navigator.clipboard.writeText(accumulatedText);
      const originalText = btnCopy.textContent;
      btnCopy.textContent = 'Copied';
      setTimeout(() => {
        btnCopy.textContent = originalText;
      }, 1500);
    } catch (e) {
      console.error('Clipboard copy failed:', e);
    }
  });

  // Fetch System Telemetry
  async function loadSystemInfo() {
    try {
      const res = await fetch('/api/info');
      if (res.ok) {
        const data = await res.json();
        if (data.model) {
          deviceVal.textContent = data.model.device || 'CPU';
          if (data.model.parameters) {
            paramsVal.textContent = `${data.model.parameters.toLocaleString()} params`;
          }
        }
      }
    } catch (e) {
      deviceVal.textContent = 'Online';
    }
  }
  loadSystemInfo();

  // Inference Execution
  async function runInference() {
    const prompt = promptInput.value;
    if (!prompt) return;

    btnGenerate.disabled = true;
    btnStop.disabled = false;

    accumulatedText = prompt;
    outputText.textContent = prompt;
    tokenRecords = [];
    tokenGrid.innerHTML = '';

    activeAbort = new AbortController();
    const startTime = performance.now();
    let tokenCount = 0;

    const payload = {
      prompt: prompt,
      max_tokens: parseInt(maxtokNum.value, 10),
      temperature: parseFloat(tempNum.value),
      top_k: parseInt(topkNum.value, 10),
      top_p: parseFloat(toppNum.value),
      seed: seedInput.value.trim() ? parseInt(seedInput.value.trim(), 10) : null
    };

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: activeAbort.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            if (!dataStr) continue;
            try {
              const msg = JSON.parse(dataStr);
              if (msg.token !== undefined) {
                tokenCount++;
                accumulatedText += msg.token;
                outputText.textContent = accumulatedText;
                tokenRecords.push({ token: msg.token, id: msg.id });

                const elapsed = (performance.now() - startTime) / 1000;
                statTokens.textContent = `${tokenCount} tokens`;
                statLatency.textContent = `${elapsed.toFixed(2)}s`;
                statSpeed.textContent = `${(tokenCount / Math.max(elapsed, 0.05)).toFixed(1)} tok/s`;

                stageText.scrollTop = stageText.scrollHeight;
              }
            } catch (jsonErr) {
              console.error('Failed to parse SSE payload', jsonErr);
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        outputText.textContent += `\n[Inference error: ${err.message}]`;
      }
    } finally {
      btnGenerate.disabled = false;
      btnStop.disabled = true;
      activeAbort = null;
      if (btnViewTokens.classList.contains('active')) {
        renderTokens();
      }
    }
  }

  btnGenerate.addEventListener('click', runInference);

  btnStop.addEventListener('click', () => {
    if (activeAbort) {
      activeAbort.abort();
    }
  });

  // Ctrl+Enter / Cmd+Enter Shortcut
  promptInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!btnGenerate.disabled) {
        runInference();
      }
    }
  });
});
