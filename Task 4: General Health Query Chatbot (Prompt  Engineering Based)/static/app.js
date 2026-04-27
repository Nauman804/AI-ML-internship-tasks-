/* ══════════════════════════════════════════════════════════
   MEDIBOT — FRONTEND LOGIC v2.0
   Features: Canvas background · Chat · BMI · Symptom checker
             Voice input · Export · Session timer · Toast
══════════════════════════════════════════════════════════ */

/* ── DOM refs ── */
const msgsScroll   = document.getElementById('msgsScroll');
const msgList      = document.getElementById('msgList');
const welcomeScreen= document.getElementById('welcomeScreen');
const typingRow    = document.getElementById('typingRow');
const userInput    = document.getElementById('userInput');
const sendBtn      = document.getElementById('sendBtn');
const clearBtn     = document.getElementById('clearBtn');
const newChatBtn   = document.getElementById('newChatBtn');
const charCount    = document.getElementById('charCount');
const msgCountEl   = document.getElementById('msgCount');
const sessionTimeEl= document.getElementById('sessionTime');
const modeLabel    = document.getElementById('modeLabel');
const bmiResult    = document.getElementById('bmiResult');
const bmiBar       = document.getElementById('bmiBar');
const calcBmi      = document.getElementById('calcBmi');
const symCheckBtn  = document.getElementById('symCheckBtn');
const scrollBtn    = document.getElementById('scrollBtn');
const voiceBtn     = document.getElementById('voiceBtn');
const exportBtn    = document.getElementById('exportBtn');
const toastEl      = document.getElementById('toast');
const sidebar      = document.getElementById('sidebar');
const menuToggle   = document.getElementById('menuToggle');

/* ── State ── */
let conversationHistory = [];
let msgCount   = 0;
let currentMode= 'general';
let sessionStart = Date.now();
let sessionInterval;
let isLoading  = false;
let recognition= null;

/* ════════════════════════════════════════
   CANVAS BACKGROUND ANIMATION
════════════════════════════════════════ */
(function initCanvas() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const COLORS = ['#00d2b8','#0095ff','#7c5cfc','#ff6b9d'];
  const NUM = 55;

  for (let i = 0; i < NUM; i++) {
    particles.push({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 2.4 + 0.5,
      dx: (Math.random() - 0.5) * 0.35,
      dy: (Math.random() - 0.5) * 0.35,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: Math.random() * 0.5 + 0.15,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0,210,184,${0.06 * (1 - dist / 130)})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + Math.round(p.alpha * 255).toString(16).padStart(2, '0');
      ctx.fill();

      p.x += p.dx;
      p.y += p.dy;
      if (p.x < 0 || p.x > W) p.dx *= -1;
      if (p.y < 0 || p.y > H) p.dy *= -1;
    });

    requestAnimationFrame(draw);
  }
  draw();
})();

/* ════════════════════════════════════════
   SESSION TIMER
════════════════════════════════════════ */
sessionInterval = setInterval(() => {
  const mins = Math.floor((Date.now() - sessionStart) / 60000);
  sessionTimeEl.textContent = mins < 60 ? `${mins}m` : `${Math.floor(mins/60)}h`;
}, 10000);

/* ════════════════════════════════════════
   TEXTAREA AUTO-RESIZE + CHAR COUNT
════════════════════════════════════════ */
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
  const len = userInput.value.length;
  charCount.textContent = len;
  charCount.className = len > 550 ? 'over' : '';
});

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

/* ════════════════════════════════════════
   MODE TOGGLE
════════════════════════════════════════ */
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    const labels = {general:'General Health Mode · Online', symptoms:'Symptoms Mode · Online', meds:'Medications Mode · Online'};
    modeLabel.textContent = labels[currentMode] || 'Online';
    showToast(`Switched to ${btn.textContent} mode`, 'success');
  });
});

/* ════════════════════════════════════════
   QUICK BUTTONS
════════════════════════════════════════ */
document.querySelectorAll('.quick-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    injectAndSend(btn.dataset.q);
    if (window.innerWidth < 820) sidebar.classList.remove('open');
  });
});

/* Suggestion cards on welcome */
document.querySelectorAll('.w-sug-card').forEach(card => {
  card.addEventListener('click', () => injectAndSend(card.dataset.q));
});

/* Toolbar topic injection */
document.querySelectorAll('.tool-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const prefix = btn.dataset.inject || '';
    userInput.focus();
    if (!userInput.value.startsWith(prefix)) {
      userInput.value = prefix + userInput.value;
      userInput.dispatchEvent(new Event('input'));
    }
    btn.classList.add('active');
    setTimeout(() => btn.classList.remove('active'), 600);
  });
});

function injectAndSend(text) {
  userInput.value = text;
  userInput.dispatchEvent(new Event('input'));
  sendMessage();
}

/* ════════════════════════════════════════
   SEND MESSAGE FLOW
════════════════════════════════════════ */
sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isLoading) return;

  // Hide welcome
  if (welcomeScreen) welcomeScreen.style.display = 'none';

  appendMessage('user', text);
  conversationHistory.push({ role: 'user', content: text });
  msgCount++;
  msgCountEl.textContent = msgCount;

  userInput.value = '';
  userInput.style.height = 'auto';
  charCount.textContent = '0';
  isLoading = true;
  sendBtn.disabled = true;

  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: conversationHistory.slice(0, -1),
        mode: currentMode,
      }),
    });

    const data = await res.json();
    hideTyping();

    if (data.error) {
      appendMessage('bot', '⚠️ Something went wrong. Please try again.', true);
      return;
    }

    appendMessage('bot', data.reply, data.safe_flag);
    conversationHistory.push({ role: 'assistant', content: data.reply });
    msgCount++;
    msgCountEl.textContent = msgCount;

  } catch (err) {
    hideTyping();
    appendMessage('bot', '⚠️ Network error. Please check your connection and try again.', true);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

/* ════════════════════════════════════════
   APPEND MESSAGE BUBBLE
════════════════════════════════════════ */
function appendMessage(role, text, isWarn = false) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  if (role === 'bot') {
    const av = document.createElement('div');
    av.className = 'avatar';
    av.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor">
      <rect x="9.5" y="2" width="5" height="20" rx="2.5"/>
      <rect x="2" y="9.5" width="20" height="5" rx="2.5"/>
    </svg>`;
    row.appendChild(av);
  }

  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}${isWarn ? ' warn' : ''}`;
  bubble.innerHTML = formatMarkdown(text);

  const footer = document.createElement('div');
  footer.className = 'bubble-footer';

  const time = document.createElement('span');
  time.className = 'btime';
  time.textContent = nowTime();
  footer.appendChild(time);

  if (role === 'bot') {
    // Copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'bubble-action';
    copyBtn.title = 'Copy response';
    copyBtn.textContent = '📋 Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(text).then(() => showToast('Copied!', 'success'));
    });
    footer.appendChild(copyBtn);

    // Feedback
    const fbRow = document.createElement('div');
    fbRow.className = 'feedback-row';

    const upBtn = document.createElement('button');
    upBtn.className = 'fb-btn up';
    upBtn.title = 'Helpful';
    upBtn.textContent = '👍';
    upBtn.addEventListener('click', () => {
      upBtn.classList.add('voted');
      downBtn.classList.add('voted');
      showToast('Thanks for the feedback!', 'success');
    });

    const downBtn = document.createElement('button');
    downBtn.className = 'fb-btn down';
    downBtn.title = 'Not helpful';
    downBtn.textContent = '👎';
    downBtn.addEventListener('click', () => {
      upBtn.classList.add('voted');
      downBtn.classList.add('voted');
      showToast('Feedback noted, sorry about that.', '');
    });

    fbRow.appendChild(upBtn);
    fbRow.appendChild(downBtn);
    footer.appendChild(fbRow);
  }

  wrap.appendChild(bubble);
  wrap.appendChild(footer);
  row.appendChild(wrap);
  msgList.appendChild(row);

  scrollToBottom();
}

/* ════════════════════════════════════════
   MARKDOWN FORMATTER
════════════════════════════════════════ */
function formatMarkdown(text) {
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/`(.+?)`/g,'<code>$1</code>')
    .replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/\n\n/g,'</p><p>')
    .replace(/^- (.+)$/gm,'<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g,m=>`<ul>${m}</ul>`)
    .replace(/\n/g,'<br>');
}

/* ════════════════════════════════════════
   TYPING INDICATOR
════════════════════════════════════════ */
function showTyping() {
  typingRow.classList.remove('hidden');
  scrollToBottom();
}
function hideTyping() {
  typingRow.classList.add('hidden');
}

/* ════════════════════════════════════════
   SCROLL MANAGEMENT
════════════════════════════════════════ */
function scrollToBottom() {
  setTimeout(() => {
    msgsScroll.scrollTop = msgsScroll.scrollHeight;
  }, 60);
}

msgsScroll.addEventListener('scroll', () => {
  const distFromBottom = msgsScroll.scrollHeight - msgsScroll.scrollTop - msgsScroll.clientHeight;
  scrollBtn.classList.toggle('visible', distFromBottom > 120);
});

scrollBtn.addEventListener('click', scrollToBottom);

/* ════════════════════════════════════════
   CLEAR / NEW CHAT
════════════════════════════════════════ */
clearBtn.addEventListener('click', resetChat);
newChatBtn.addEventListener('click', () => {
  resetChat();
  sessionStart = Date.now();
  msgCount = 0;
  msgCountEl.textContent = '0';
  sessionTimeEl.textContent = '0m';
  showToast('New chat started!', 'success');
});

function resetChat() {
  msgList.innerHTML = '';
  conversationHistory = [];
  if (welcomeScreen) welcomeScreen.style.display = 'flex';
}

/* ════════════════════════════════════════
   MOBILE SIDEBAR TOGGLE
════════════════════════════════════════ */
menuToggle.addEventListener('click', () => {
  sidebar.classList.toggle('open');
});
document.addEventListener('click', e => {
  if (sidebar.classList.contains('open') &&
      !sidebar.contains(e.target) &&
      !menuToggle.contains(e.target)) {
    sidebar.classList.remove('open');
  }
});

/* ════════════════════════════════════════
   BMI CALCULATOR
════════════════════════════════════════ */
calcBmi.addEventListener('click', calculateBMI);
document.getElementById('bmiWeight').addEventListener('keydown', e => e.key === 'Enter' && calculateBMI());
document.getElementById('bmiHeight').addEventListener('keydown', e => e.key === 'Enter' && calculateBMI());

function calculateBMI() {
  const w = parseFloat(document.getElementById('bmiWeight').value);
  const h = parseFloat(document.getElementById('bmiHeight').value) / 100;

  if (!w || !h || w <= 0 || h <= 0) {
    showToast('Please enter valid weight and height.', 'error');
    return;
  }

  const bmi = w / (h * h);
  const bmiFixed = bmi.toFixed(1);
  let cat, catClass, barW, barColor;

  if (bmi < 18.5) {
    cat = 'Underweight'; catClass = 'underweight'; barW = 20; barColor = '#0095ff';
  } else if (bmi < 25) {
    cat = 'Normal Weight'; catClass = 'normal'; barW = 45; barColor = '#00e676';
  } else if (bmi < 30) {
    cat = 'Overweight'; catClass = 'overweight'; barW = 68; barColor = '#ff9800';
  } else {
    cat = 'Obese'; catClass = 'obese'; barW = 90; barColor = '#ff4d6d';
  }

  bmiResult.innerHTML = `
    <span class="bmi-num">${bmiFixed}</span>
    <span class="bmi-cat ${catClass}">${cat}</span>
  `;
  bmiBar.style.width = barW + '%';
  bmiBar.style.background = barColor;

  showToast(`BMI: ${bmiFixed} — ${cat}`, catClass === 'normal' ? 'success' : '');
}

/* ════════════════════════════════════════
   SYMPTOM CHECKER
════════════════════════════════════════ */
const selectedSymptoms = new Set();

document.querySelectorAll('.sym-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const sym = chip.dataset.sym;
    if (selectedSymptoms.has(sym)) {
      selectedSymptoms.delete(sym);
      chip.classList.remove('selected');
    } else {
      selectedSymptoms.add(sym);
      chip.classList.add('selected');
    }
    const count = selectedSymptoms.size;
    symCheckBtn.textContent = count
      ? `Ask MediBot about ${count} symptom${count > 1 ? 's' : ''} →`
      : 'Ask MediBot →';
  });
});

symCheckBtn.addEventListener('click', () => {
  if (selectedSymptoms.size === 0) {
    showToast('Please select at least one symptom.', 'error');
    return;
  }
  const symList = Array.from(selectedSymptoms).join(', ');
  const q = `I have the following symptoms: ${symList}. What could be the possible causes, and what should I do?`;
  injectAndSend(q);

  // Reset
  selectedSymptoms.clear();
  document.querySelectorAll('.sym-chip').forEach(c => c.classList.remove('selected'));
  symCheckBtn.textContent = 'Ask MediBot →';

  if (window.innerWidth < 820) sidebar.classList.remove('open');
});

/* ════════════════════════════════════════
   VOICE INPUT (Web Speech API)
════════════════════════════════════════ */
voiceBtn.addEventListener('click', () => {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    showToast('Voice not supported in this browser.', 'error');
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (recognition) {
    recognition.stop();
    recognition = null;
    voiceBtn.textContent = '🎤';
    voiceBtn.classList.remove('active');
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    voiceBtn.textContent = '⏹';
    voiceBtn.classList.add('active');
    showToast('Listening… speak now', '');
  };

  recognition.onresult = e => {
    const transcript = e.results[0][0].transcript;
    userInput.value = transcript;
    userInput.dispatchEvent(new Event('input'));
    voiceBtn.textContent = '🎤';
    voiceBtn.classList.remove('active');
    recognition = null;
  };

  recognition.onerror = () => {
    showToast('Voice input failed. Try again.', 'error');
    voiceBtn.textContent = '🎤';
    voiceBtn.classList.remove('active');
    recognition = null;
  };

  recognition.onend = () => {
    if (recognition) {
      voiceBtn.textContent = '🎤';
      voiceBtn.classList.remove('active');
      recognition = null;
    }
  };

  recognition.start();
});

/* ════════════════════════════════════════
   EXPORT CHAT
════════════════════════════════════════ */
exportBtn.addEventListener('click', () => {
  if (conversationHistory.length === 0) {
    showToast('Nothing to export yet.', 'error');
    return;
  }
  const lines = conversationHistory.map(m => {
    const role = m.role === 'user' ? 'You' : 'MediBot';
    return `[${role}]\n${m.content}\n`;
  });
  const content = `MediBot Chat Export\nDate: ${new Date().toLocaleString()}\n${'─'.repeat(40)}\n\n${lines.join('\n')}`;
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `medibot-chat-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Chat exported!', 'success');
});

/* ════════════════════════════════════════
   TOAST
════════════════════════════════════════ */
let toastTimer;
function showToast(msg, type = '') {
  clearTimeout(toastTimer);
  toastEl.textContent = msg;
  toastEl.className = `toast show${type ? ' ' + type : ''}`;
  toastTimer = setTimeout(() => {
    toastEl.classList.remove('show');
  }, 2600);
}

/* ════════════════════════════════════════
   TIME HELPER
════════════════════════════════════════ */
function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/* ── Focus input on load ── */
setTimeout(() => userInput.focus(), 400);