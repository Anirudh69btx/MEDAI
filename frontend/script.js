/**
 * MindCare AI — Conversational Frontend Assistant
 * Pure Vanilla JavaScript ES6+ (No Frameworks)
 * Handles sequential assessment flow, API communication, and dynamic clinical result rendering.
 */

(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Configuration & State
  // ---------------------------------------------------------------------------
  const API_BASE_URL = window.location.origin.includes(':5000') 
    ? window.location.origin 
    : 'http://127.0.0.1:5000';

  const COUNTRIES = [
    'United States', 'United Kingdom', 'Canada', 'India', 'Australia', 'Germany',
    'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Colombia', 'Costa Rica', 
    'Croatia', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Georgia', 
    'Greece', 'Ireland', 'Israel', 'Italy', 'Mexico', 'Moldova', 'Netherlands', 
    'New Zealand', 'Nigeria', 'Philippines', 'Poland', 'Portugal', 'Russia', 
    'Singapore', 'South Africa', 'Sweden', 'Switzerland', 'Thailand'
  ];

  // 15 Raw Features Questionnaire Sequence
  const QUESTIONS = [
    {
      id: 1,
      field: 'gender',
      title: 'Gender Identity',
      question: "Let's begin your assessment. What is your biological sex or gender identity?",
      options: ['Male', 'Female'],
      helper: 'Select your gender identity to calibrate demographic baseline.'
    },
    {
      id: 2,
      field: 'country',
      title: 'Country of Residence',
      question: 'What country are you currently residing in?',
      type: 'country',
      options: ['United States', 'United Kingdom', 'Canada', 'India', 'Australia', 'Germany'],
      helper: 'Choose from common regions or select from the full list.'
    },
    {
      id: 3,
      field: 'occupation',
      title: 'Primary Occupation',
      question: 'What is your current occupation or primary daily activity?',
      options: ['Corporate', 'Student', 'Business', 'Housewife', 'Others'],
      helper: 'Occupational environments correlate with specific stress factors.'
    },
    {
      id: 4,
      field: 'self_employed',
      title: 'Employment Type',
      question: 'Are you currently self-employed or running your own venture?',
      options: ['No', 'Yes'],
      helper: 'Self-employment status helps evaluate workload autonomy.'
    },
    {
      id: 5,
      field: 'family_history',
      title: 'Family History',
      question: 'Do you have a known family history of mental health illness?',
      options: ['No', 'Yes'],
      helper: 'Genetic and familial predispositions are vital risk markers.'
    },
    {
      id: 6,
      field: 'days_indoors',
      title: 'Indoors Duration',
      question: 'On average, how much consecutive time do you spend indoors without going outside?',
      options: ['Go out Every day', '1-14 days', '15-30 days', '31-60 days', 'More than 2 months'],
      helper: 'Physical isolation and lack of sunlight strongly affect mood regulation.'
    },
    {
      id: 7,
      field: 'growing_stress',
      title: 'Growing Stress',
      question: 'Have you been experiencing noticeably escalating stress or pressure lately?',
      options: ['No', 'Maybe', 'Yes'],
      helper: 'Evaluate your felt stress level over the past few weeks.'
    },
    {
      id: 8,
      field: 'changes_habits',
      title: 'Habit Changes',
      question: 'Have you noticed uncharacteristic shifts in your eating, sleeping, or personal hygiene habits?',
      options: ['No', 'Maybe', 'Yes'],
      helper: 'Disruptions in circadian rhythms or appetite often indicate distress.'
    },
    {
      id: 9,
      field: 'mental_health_history',
      title: 'Mental Health History',
      question: 'Do you have a personal prior history or clinical diagnosis of mental health conditions?',
      options: ['No', 'Maybe', 'Yes'],
      helper: 'Personal clinical baseline enhances predictive accuracy.'
    },
    {
      id: 10,
      field: 'mood_swings',
      title: 'Mood Swings',
      question: 'How would you describe the frequency and intensity of your mood swings?',
      options: ['Low', 'Medium', 'High'],
      helper: 'Emotional volatility is an important affective indicator.'
    },
    {
      id: 11,
      field: 'coping_struggles',
      title: 'Coping Struggles',
      question: 'Do you currently find yourself struggling to cope with everyday emotional challenges?',
      options: ['No', 'Yes'],
      helper: 'Assesses active resilience and emotional exhaustion.'
    },
    {
      id: 12,
      field: 'work_interest',
      title: 'Work / Study Interest',
      question: 'Have you experienced anhedonia or a loss of interest and focus in your work or studies?',
      options: ['No', 'Maybe', 'Yes'],
      helper: 'Loss of interest (anhedonia) is a hallmark clinical symptom.'
    },
    {
      id: 13,
      field: 'social_weakness',
      title: 'Social Fatigue / Detachment',
      question: 'Do you feel social withdrawal, weakness, or increased detachment from family and peers?',
      options: ['No', 'Maybe', 'Yes'],
      helper: 'Measures interpersonal connection and social support resilience.'
    },
    {
      id: 14,
      field: 'mental_health_interview',
      title: 'Interview Openness',
      question: 'Would you feel willing to discuss mental health challenges in an employment or health interview?',
      options: ['No', 'Maybe', 'Yes'],
      helper: 'Reflects perceived stigma and communication openness.'
    },
    {
      id: 15,
      field: 'care_options',
      title: 'Care Awareness',
      question: 'Lastly, are you aware of and do you have access to mental health care options or support programs?',
      options: ['No', 'Not sure', 'Yes'],
      helper: 'Evaluates awareness of local resources, therapy, or EAP benefits.'
    }
  ];

  // Verified Static Metrics Benchmark Cache (fallback if /api/metrics is unreachable)
  const BENCHMARK_METRICS = {
    production: {
      model_name: 'PyTorch Deep Learning',
      accuracy: '77.78%',
      recall: '83.97%',
      f1_score: '79.24%',
      roc_auc: '86.93%',
      precision: '75.01%'
    },
    baseline: {
      model_name: 'Decision Tree Classifier',
      accuracy: '75.84%',
      recall: '82.34%',
      f1_score: '77.48%',
      roc_auc: '84.40%',
      precision: '73.16%'
    }
  };

  // State
  let currentStep = 0; // 0 = welcome/ready, 1..15 = active questions, 16 = finished
  const userResponses = {};
  let isSubmitting = false;
  let serverOnline = false;

  // DOM Elements
  const chatMessages = document.getElementById('chat-messages');
  const quickOptions = document.getElementById('quick-options');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const serverStatus = document.getElementById('server-status');
  const progressFill = document.getElementById('progress-fill');
  const progressStepText = document.getElementById('progress-step-text');
  const progressPercentText = document.getElementById('progress-percent-text');
  const btnRestart = document.getElementById('btn-restart');
  const btnMetricsModal = document.getElementById('btn-metrics-modal');
  const metricsModal = document.getElementById('metrics-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const btnModalDismiss = document.getElementById('btn-modal-dismiss');

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------
  function init() {
    setupEventListeners();
    checkServerHealth();
    startConversation();
  }

  // ---------------------------------------------------------------------------
  // Event Listeners
  // ---------------------------------------------------------------------------
  function setupEventListeners() {
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      handleFormSubmit();
    });

    btnRestart.addEventListener('click', () => {
      resetChat();
    });

    btnMetricsModal.addEventListener('click', () => {
      openModal();
    });

    btnCloseModal.addEventListener('click', () => {
      closeModal();
    });

    btnModalDismiss.addEventListener('click', () => {
      closeModal();
    });

    metricsModal.addEventListener('click', (e) => {
      if (e.target === metricsModal) {
        closeModal();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && metricsModal.classList.contains('active')) {
        closeModal();
      }
    });
  }

  function openModal() {
    metricsModal.classList.add('active');
    metricsModal.setAttribute('aria-hidden', 'false');
  }

  function closeModal() {
    metricsModal.classList.remove('active');
    metricsModal.setAttribute('aria-hidden', 'true');
  }

  // ---------------------------------------------------------------------------
  // Server Health Check
  // ---------------------------------------------------------------------------
  async function checkServerHealth() {
    updateStatusIndicator('checking', 'CHECKING...');
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        serverOnline = true;
        updateStatusIndicator('online', 'AI SERVER ONLINE');
      } else {
        serverOnline = false;
        updateStatusIndicator('offline', 'API OFFLINE');
      }
    } catch (err) {
      serverOnline = false;
      updateStatusIndicator('offline', 'SERVER OFFLINE');
      console.warn('[MindCare AI] Health check failed:', err);
    }
  }

  function updateStatusIndicator(state, text) {
    serverStatus.className = `status-indicator ${state}`;
    serverStatus.querySelector('.status-label').textContent = text;
    serverStatus.title = `Flask Backend: ${API_BASE_URL} (${text})`;
  }

  // ---------------------------------------------------------------------------
  // Conversational Flow Management
  // ---------------------------------------------------------------------------
  function startConversation() {
    currentStep = 0;
    clearObject(userResponses);
    chatMessages.innerHTML = '';
    quickOptions.innerHTML = '';
    updateProgress(0);

    // Initial AI Welcome Greeting
    addAiMessage(`
      <p><strong>Hello! I'm MindCare AI</strong> — your conversational mental health monitoring and personalized recommendation assistant.</p>
      <p>I will guide you through a standardized <strong>15-question clinical evaluation</strong>. At the end, our deep neural network will analyze your answers and generate structured recommendations.</p>
      <p style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.4rem;"><em>Click "Start Assessment" below or type any message to begin.</em></p>
    `);

    renderOptions([
      { label: '✨ Start Assessment', value: '__start__' }
    ]);
  }

  function advanceToStep(stepNumber) {
    currentStep = stepNumber;
    updateProgress(stepNumber);

    if (stepNumber > QUESTIONS.length) {
      // Completed all 15 questions -> Trigger prediction
      quickOptions.innerHTML = '';
      userInput.placeholder = 'Analyzing assessment...';
      userInput.disabled = true;
      btnSend.disabled = true;
      submitPrediction();
      return;
    }

    const q = QUESTIONS[stepNumber - 1];
    
    // Add AI Question Message
    setTimeout(() => {
      addAiMessage(`
        <p><strong>${q.question}</strong></p>
        <p style="color: var(--text-muted); font-size: 0.78rem; margin-top: 0.25rem;">${q.helper}</p>
      `, `Step ${q.id} of 15`);

      renderQuestionControls(q);
      userInput.disabled = false;
      btnSend.disabled = false;
      userInput.focus();
    }, 280);
  }

  function renderQuestionControls(question) {
    quickOptions.innerHTML = '';

    if (question.type === 'country') {
      // Render quick country chips + searchable select dropdown
      const wrapper = document.createElement('div');
      wrapper.className = 'country-select-wrapper';

      const select = document.createElement('select');
      select.className = 'country-select';
      select.setAttribute('aria-label', 'Select country');
      select.innerHTML = `<option value="" disabled selected>Or choose all countries (${COUNTRIES.length})...</option>` +
        COUNTRIES.map(c => `<option value="${c}">${c}</option>`).join('');

      select.addEventListener('change', () => {
        if (select.value) {
          handleUserAnswer(select.value);
        }
      });

      // Quick popular country chips
      question.options.forEach(opt => {
        const chip = createChip(opt, opt, () => handleUserAnswer(opt));
        quickOptions.appendChild(chip);
      });

      quickOptions.appendChild(select);
    } else {
      // Standard option chips
      question.options.forEach(opt => {
        const chip = createChip(opt, opt, () => handleUserAnswer(opt));
        quickOptions.appendChild(chip);
      });
    }

    userInput.placeholder = `Select an option above or type your answer...`;
  }

  function createChip(label, value, onClick) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'option-chip';
    chip.textContent = label;
    chip.addEventListener('click', onClick);
    return chip;
  }

  // ---------------------------------------------------------------------------
  // Input Handling
  // ---------------------------------------------------------------------------
  function handleFormSubmit() {
    const val = userInput.value.trim();
    if (!val || isSubmitting) return;

    userInput.value = '';

    if (currentStep === 0) {
      advanceToStep(1);
      return;
    }

    handleUserAnswer(val);
  }

  function handleUserAnswer(answer) {
    if (isSubmitting) return;

    if (answer === '__start__') {
      advanceToStep(1);
      return;
    }

    const activeQuestion = QUESTIONS[currentStep - 1];
    if (!activeQuestion) return;

    // Normalize and validate answer against acceptable options
    const matchedValue = normalizeOptionMatch(answer, activeQuestion);

    // Record response
    userResponses[activeQuestion.field] = matchedValue;

    // Display user message in chat
    addUserMessage(matchedValue);

    // Disable input momentarily while AI advances
    quickOptions.innerHTML = '';
    userInput.disabled = true;
    btnSend.disabled = true;

    // Advance to next question
    advanceToStep(currentStep + 1);
  }

  function normalizeOptionMatch(input, question) {
    const clean = input.trim();
    const cleanLower = clean.toLowerCase();

    // If it's country, check case-insensitive match in COUNTRIES
    if (question.type === 'country') {
      const match = COUNTRIES.find(c => c.toLowerCase() === cleanLower);
      if (match) return match;
      return clean; // Fallback to raw string
    }

    // Check exact or partial match in question options
    for (const opt of question.options) {
      if (opt.toLowerCase() === cleanLower) return opt;
    }

    // Common abbreviations
    if (cleanLower === 'y' || cleanLower === 'yes' || cleanLower === 'true') {
      const optYes = question.options.find(o => o.toLowerCase() === 'yes');
      if (optYes) return optYes;
    }
    if (cleanLower === 'n' || cleanLower === 'no' || cleanLower === 'false') {
      const optNo = question.options.find(o => o.toLowerCase() === 'no');
      if (optNo) return optNo;
    }
    if (cleanLower === 'm' || cleanLower === 'maybe') {
      const optMaybe = question.options.find(o => o.toLowerCase() === 'maybe');
      if (optMaybe) return optMaybe;
    }

    // Return first option as graceful fallback or exact input if none matched
    return question.options.find(o => o.toLowerCase().startsWith(cleanLower)) || clean;
  }

  // ---------------------------------------------------------------------------
  // Message Rendering in Chat
  // ---------------------------------------------------------------------------
  function addAiMessage(htmlContent, metaBadge = null) {
    const row = document.createElement('div');
    row.className = 'message-row ai';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a8 8 0 0 0-8 8c0 3.3 2 6.2 5 7.4V20a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-2.6c3-1.2 5-4.1 5-7.4a8 8 0 0 0-8-8z"/>
      </svg>
    `;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const text = document.createElement('div');
    text.className = 'msg-text';
    text.innerHTML = htmlContent;
    bubble.appendChild(text);

    if (metaBadge) {
      const meta = document.createElement('div');
      meta.className = 'msg-meta';
      meta.innerHTML = `<span class="msg-badge-step">${metaBadge}</span> &bull; MindCare Clinical AI`;
      bubble.appendChild(meta);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function addUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const msgText = document.createElement('div');
    msgText.className = 'msg-text';
    msgText.textContent = text;
    bubble.appendChild(msgText);

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = 'You';
    bubble.appendChild(meta);

    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function showTypingIndicator(message = 'MindCare AI is analyzing with PyTorch Neural Network...') {
    removeTypingIndicator();

    const row = document.createElement('div');
    row.id = 'typing-indicator-row';
    row.className = 'message-row ai';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
        <circle cx="12" cy="12" r="10"/>
      </svg>
    `;

    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = `
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
      <span>${escapeHtml(message)}</span>
    `;

    row.appendChild(avatar);
    row.appendChild(indicator);
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator-row');
    if (el) el.remove();
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function updateProgress(stepNumber) {
    const total = QUESTIONS.length;
    const clamped = Math.min(Math.max(stepNumber, 0), total);
    const percent = Math.round((clamped / total) * 100);

    progressFill.style.width = `${percent}%`;
    progressPercentText.textContent = `${percent}%`;

    if (clamped === 0) {
      progressStepText.textContent = 'Ready to start';
    } else if (clamped > total) {
      progressStepText.textContent = 'Assessment Completed';
      progressFill.style.width = '100%';
      progressPercentText.textContent = '100%';
    } else {
      progressStepText.textContent = `Question ${clamped} of ${total}`;
    }
  }

  // ---------------------------------------------------------------------------
  // Backend Prediction Submission
  // ---------------------------------------------------------------------------
  async function submitPrediction() {
    isSubmitting = true;
    showTypingIndicator('Running PyTorch 77-feature neural inference...');

    // Log the exact 15 raw feature payload being submitted
    console.log('[MindCare AI] Submitting 15 raw features to Flask backend:', userResponses);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(userResponses)
      });

      removeTypingIndicator();

      if (!response.ok) {
        let errorMsg = `Server returned HTTP ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.error) errorMsg = errData.error;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const resultData = await response.json();
      console.log('[MindCare AI] Real Prediction Received from Backend:', resultData);

      if (!resultData.success && !resultData.prediction) {
        throw new Error(resultData.error || 'Prediction calculation failed.');
      }

      // Render the result
      renderPredictionResult(resultData);

    } catch (err) {
      removeTypingIndicator();
      console.error('[MindCare AI] Prediction Request Error:', err);
      handleApiError(err.message);
    } finally {
      isSubmitting = false;
      userInput.disabled = false;
      btnSend.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Result Display Rendering
  // ---------------------------------------------------------------------------
  function renderPredictionResult(apiResponse) {
    const pred = apiResponse.prediction || apiResponse.result || {};
    const rec = pred.recommendation || {};

    const predictedClass = String(pred.predicted_class || 'Unknown').trim();
    const isTreatmentNeeded = predictedClass.toLowerCase() === 'yes';
    const confidenceVal = Number(pred.confidence || 0);
    const confidencePct = (confidenceVal * 100).toFixed(2);
    const modelUsed = pred.model_used || 'PyTorch Deep Learning';
    const riskLevel = rec.risk_level || (isTreatmentNeeded ? 'moderate_to_high' : 'minimal_to_low');

    // Probabilities
    const probs = pred.probabilities || {};
    const probNo = Number(probs.No !== undefined ? probs.No : (1 - confidenceVal));
    const probYes = Number(probs.Yes !== undefined ? probs.Yes : confidenceVal);
    const probNoPct = (probNo * 100).toFixed(2);
    const probYesPct = (probYes * 100).toFixed(2);

    // Format risk level string
    const formattedRisk = riskLevel.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const riskPillClass = isTreatmentNeeded ? 'moderate' : 'low';

    // Lists
    const immediateActions = rec.immediate_actions || [];
    const selfCare = rec.self_care || [];
    const professionalReferral = rec.professional_referral || [];
    const resources = rec.resources || [];
    const disclaimer = rec.disclaimer || 'DISCLAIMER: These recommendations are for informational purposes only and do not constitute medical advice.';

    // Create Result Card Container
    const row = document.createElement('div');
    row.className = 'message-row ai';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
      </svg>
    `;

    const card = document.createElement('div');
    card.className = 'result-card';

    card.innerHTML = `
      <div class="result-header">
        <div class="result-title-group">
          <h3>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2a8 8 0 0 0-8 8c0 3.3 2 6.2 5 7.4V20a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-2.6c3-1.2 5-4.1 5-7.4a8 8 0 0 0-8-8z"/>
            </svg>
            MINDCARE AI CLINICAL ANALYSIS
          </h3>
          <div class="result-timestamp">Inference Model: <strong>${escapeHtml(modelUsed)}</strong> &bull; ${new Date().toLocaleTimeString()}</div>
        </div>
      </div>

      <!-- Summary Tiles -->
      <div class="prediction-summary-grid">
        <div class="summary-tile">
          <span class="tile-label">Predicted Status</span>
          <span class="tile-value ${isTreatmentNeeded ? 'status-yes' : 'status-no'}">
            ${isTreatmentNeeded ? '● ATTENTION RECOMMENDED' : '● MINIMAL RISK'}
          </span>
          <span style="font-size: 0.72rem; color: var(--text-muted);">${isTreatmentNeeded ? 'Support recommended' : 'Routine wellness indicated'}</span>
        </div>

        <div class="summary-tile">
          <span class="tile-label">Prediction Confidence</span>
          <span class="tile-value">${confidencePct}%</span>
          <span style="font-size: 0.72rem; color: var(--text-muted);">Posterior Probability</span>
        </div>

        <div class="summary-tile">
          <span class="tile-label">Assessed Risk Level</span>
          <span class="risk-pill ${riskPillClass}">${escapeHtml(formattedRisk)}</span>
        </div>
      </div>

      <!-- Probability Distribution -->
      <div class="prob-section">
        <div class="prob-title">
          <span>Softmax Probability Distribution</span>
          <span>Binary Classification (No / Yes)</span>
        </div>
        <div class="prob-bars-grid">
          <div class="prob-bar-row">
            <span class="prob-bar-label">No (Minimal):</span>
            <div class="prob-track">
              <div class="prob-fill no" style="width: ${probNoPct}%;"></div>
            </div>
            <span class="prob-val-text">${probNoPct}%</span>
          </div>
          <div class="prob-bar-row">
            <span class="prob-bar-label">Yes (Attention):</span>
            <div class="prob-track">
              <div class="prob-fill yes" style="width: ${probYesPct}%;"></div>
            </div>
            <span class="prob-val-text">${probYesPct}%</span>
          </div>
        </div>
      </div>

      <!-- Verified Model Performance (Clearly separated from prediction confidence) -->
      <div class="model-verify-box">
        <div class="model-verify-title">
          <span>★ Verified Model Benchmark Performance</span>
          <span style="font-size: 0.68rem; color: var(--text-dim);">Offline Test Validation</span>
        </div>
        <p style="color: var(--text-muted); font-size: 0.72rem;">Global evaluation metrics of ${escapeHtml(modelUsed)} on 58,473 test instances:</p>
        <div class="model-metrics-inline">
          <span>Accuracy: <strong>77.78%</strong></span>
          <span>Recall: <strong>83.97%</strong></span>
          <span>F1-Score: <strong>79.24%</strong></span>
          <span>ROC-AUC: <strong>86.93%</strong></span>
          <span>Precision: <strong>75.01%</strong></span>
        </div>
      </div>

      <!-- Structured Recommendations -->
      <div class="recommendations-section">
        ${immediateActions.length > 0 ? `
          <div class="rec-group" style="border-color: rgba(239, 68, 68, 0.4);">
            <div class="rec-group-title" style="color: #f87171;">⚠️ Immediate Actions</div>
            <ul>
              ${immediateActions.map(act => `<li>${escapeHtml(act)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}

        ${selfCare.length > 0 ? `
          <div class="rec-group">
            <div class="rec-group-title">🌱 Self-Care Strategies</div>
            <ul>
              ${selfCare.map(sc => `<li>${escapeHtml(sc)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}

        ${professionalReferral.length > 0 ? `
          <div class="rec-group">
            <div class="rec-group-title">🩺 Professional Referral Guidance</div>
            <ul>
              ${professionalReferral.map(pr => `<li>${escapeHtml(pr)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}

        ${resources.length > 0 ? `
          <div class="rec-group">
            <div class="rec-group-title">📞 Helplines & Verified Resources</div>
            <ul class="resources-list">
              ${resources.map(res => `<li>${formatResourceItem(res)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
      </div>

      <!-- Medical Disclaimer -->
      <div class="disclaimer-box">
        <strong>MEDICAL DISCLAIMER:</strong> ${escapeHtml(disclaimer)}
      </div>

      <!-- Restart Action Button -->
      <button type="button" class="btn-new-analysis" id="btn-start-new-analysis">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5"/>
        </svg>
        Start New Mental Health Analysis
      </button>
    `;

    row.appendChild(avatar);
    row.appendChild(card);
    chatMessages.appendChild(row);

    // Bind restart button on the card
    const btnNew = card.querySelector('#btn-start-new-analysis');
    if (btnNew) {
      btnNew.addEventListener('click', () => resetChat());
    }

    scrollToBottom();
  }

  function formatResourceItem(resourceText) {
    // Convert URLs or phone numbers to clickable links
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const phoneRegex = /(\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b)/g;

    let formatted = escapeHtml(resourceText);
    formatted = formatted.replace(urlRegex, url => `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
    return formatted;
  }

  // ---------------------------------------------------------------------------
  // Error Handling
  // ---------------------------------------------------------------------------
  function handleApiError(errorMessage) {
    addAiMessage(`
      <p style="color: #f87171;"><strong>Assessment Server Connection Issue</strong></p>
      <p>MindCare AI could not reach the Flask backend at <code>${escapeHtml(API_BASE_URL)}</code>.</p>
      <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem;">Reason: <em>${escapeHtml(errorMessage)}</em></p>
      <p style="font-size: 0.82rem; margin-top: 0.5rem;">Please ensure the Flask backend is running on port 5000:
      <br><code style="font-family: var(--font-mono); background: rgba(255,255,255,0.08); padding: 0.1rem 0.4rem; border-radius: 4px;">python app.py</code></p>
      <div style="margin-top: 0.75rem;">
        <button type="button" class="btn-primary" id="btn-retry-predict" style="padding: 0.4rem 0.9rem; font-size: 0.8rem;">
          🔄 Retry Submission
        </button>
      </div>
    `);

    const btnRetry = document.getElementById('btn-retry-predict');
    if (btnRetry) {
      btnRetry.addEventListener('click', () => {
        btnRetry.disabled = true;
        submitPrediction();
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Reset & Helpers
  // ---------------------------------------------------------------------------
  function resetChat() {
    startConversation();
  }

  function clearObject(obj) {
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        delete obj[key];
      }
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ---------------------------------------------------------------------------
  // Start Application
  // ---------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', init);

})();
