const authConfig = {
  domain: 'YOUR_AUTH0_DOMAIN',
  clientId: 'YOUR_AUTH0_CLIENT_ID',
  audience: 'https://api.ark.cv',
  damAssignments: {
    'jordan@ark.cv': 'Lower Granite Dam',
    'maria@ark.cv': 'Bonneville Dam',
    'sam@ark.cv': 'Chief Joseph Dam',
  },
};

const demoAccount = {
  id: 'demo-operator',
  name: 'Jordan Blake',
  email: 'demo@ark.cv',
  role: 'Operator',
  damId: 'Lower Granite Dam',
  isDemo: true,
};

const damSummaries = {
  'Lower Granite Dam': { total: 12842, manual: 312, pending: 3 },
  'Bonneville Dam': { total: 15403, manual: 287, pending: 1 },
  'Chief Joseph Dam': { total: 8421, manual: 176, pending: 1 },
};

const clipQueueSeed = [
  {
    id: 'LC-2307',
    damId: 'Lower Granite Dam',
    capturedAt: '2024-03-29T09:42:00-07:00',
    location: 'Station 12B',
    reviewType: 'species',
    speciesGuess: 'Chinook AA',
    adiposeFlag: 'Clipped',
    countGuess: 1,
    clipUrl:
      'https://storage.googleapis.com/arkcv-demo-assets/fish-clip-01.mp4',
    boundingBoxes: [
      { x: 0.33, y: 0.28, width: 0.18, height: 0.25, label: 'Chinook', confidence: 86, highlight: true },
    ],
    needs: ['Species ID'],
  },
  {
    id: 'LC-2308',
    damId: 'Lower Granite Dam',
    capturedAt: '2024-03-29T10:15:00-07:00',
    location: 'Station 09A',
    reviewType: 'occlusion',
    speciesGuess: 'Steelhead AP',
    adiposeFlag: 'Present',
    countGuess: 2,
    clipUrl:
      'https://storage.googleapis.com/arkcv-demo-assets/fish-clip-02.mp4',
    boundingBoxes: [
      { x: 0.28, y: 0.32, width: 0.22, height: 0.26, label: 'Steelhead', confidence: 62, highlight: true },
      { x: 0.55, y: 0.35, width: 0.2, height: 0.24, label: 'Steelhead', confidence: 58, highlight: false },
    ],
    needs: ['Occlusion'],
  },
  {
    id: 'LC-2309',
    damId: 'Lower Granite Dam',
    capturedAt: '2024-03-29T11:02:00-07:00',
    location: 'Station 07D',
    reviewType: 'species',
    speciesGuess: 'Sockeye AP',
    adiposeFlag: 'Present',
    countGuess: 1,
    clipUrl:
      'https://storage.googleapis.com/arkcv-demo-assets/fish-clip-03.mp4',
    boundingBoxes: [
      { x: 0.38, y: 0.3, width: 0.16, height: 0.23, label: 'Sockeye', confidence: 64, highlight: true },
    ],
    needs: ['Species ID'],
  },
  {
    id: 'BV-1184',
    damId: 'Bonneville Dam',
    capturedAt: '2024-03-29T08:15:00-07:00',
    location: 'East Ladder 02',
    reviewType: 'species',
    speciesGuess: 'Coho AA',
    adiposeFlag: 'Clipped',
    countGuess: 1,
    clipUrl:
      'https://storage.googleapis.com/arkcv-demo-assets/fish-clip-04.mp4',
    boundingBoxes: [
      { x: 0.35, y: 0.33, width: 0.18, height: 0.22, label: 'Coho', confidence: 71, highlight: true },
    ],
    needs: ['Species ID'],
  },
  {
    id: 'CJ-442',
    damId: 'Chief Joseph Dam',
    capturedAt: '2024-03-29T07:58:00-07:00',
    location: 'Channel West 01',
    reviewType: 'occlusion',
    speciesGuess: 'Sockeye AP',
    adiposeFlag: 'Present',
    countGuess: 3,
    clipUrl:
      'https://storage.googleapis.com/arkcv-demo-assets/fish-clip-05.mp4',
    boundingBoxes: [
      { x: 0.26, y: 0.31, width: 0.19, height: 0.24, label: 'Sockeye', confidence: 54, highlight: true },
      { x: 0.49, y: 0.34, width: 0.2, height: 0.25, label: 'Sockeye', confidence: 51, highlight: false },
    ],
    needs: ['Occlusion'],
  },
];

const cloneClip = (clip) => ({
  ...clip,
  boundingBoxes: clip.boundingBoxes.map((box) => ({ ...box })),
  needs: [...clip.needs],
});

let clipQueue = clipQueueSeed.map(cloneClip);

const resetClipQueue = () => {
  clipQueue = clipQueueSeed.map(cloneClip);
};

const speciesOptions = [
  'Bull Trout',
  'Chinook AA',
  'Chinook AP',
  'Coho AA',
  'Coho AP',
  'Lamprey',
  'Sockeye AA',
  'Sockeye AP',
  'Steelhead AA',
  'Steelhead AP',
];

const state = {
  auth0Client: null,
  user: null,
  damId: null,
  isDemo: false,
  filters: {
    sort: 'fifo',
    reviewType: 'all',
    startDate: '',
    endDate: '',
    location: '',
  },
  queueIndex: 0,
  reviewHistory: [],
  changeLog: [],
  lastSync: null,
};

const $ = (id) => document.getElementById(id);

const sortByDateAsc = (a, b) => new Date(a.capturedAt) - new Date(b.capturedAt);
const sortByDateDesc = (a, b) => new Date(b.capturedAt) - new Date(a.capturedAt);

const formatNumber = (value) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);

const formatTimestamp = (value) =>
  new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));

const formatRetentionCountdown = (capturedAt) => {
  const retentionMs = 30 * 24 * 60 * 60 * 1000;
  const expires = new Date(capturedAt).getTime() + retentionMs;
  const diff = expires - Date.now();
  if (diff <= 0) return 'expires soon';
  const days = Math.floor(diff / (24 * 60 * 60 * 1000));
  const hours = Math.floor((diff % (24 * 60 * 60 * 1000)) / (60 * 60 * 1000));
  return `${days}d ${hours}h remaining`;
};

const showAuthError = (message) => {
  const container = $('auth-error');
  container.textContent = message;
  container.hidden = false;
};

const hideAuthError = () => {
  $('auth-error').hidden = true;
};

const isAuthConfigured = () =>
  authConfig.domain && !authConfig.domain.includes('YOUR_') && authConfig.clientId && !authConfig.clientId.includes('YOUR_');

const mapUserToDam = (user) => {
  if (user.isDemo) return user.damId;
  const email = user.email?.toLowerCase();
  return authConfig.damAssignments[email] ?? null;
};

const populateLocationFilter = (damId) => {
  const select = $('filter-location');
  select.innerHTML = '<option value="">All</option>';
  const locations = Array.from(
    new Set(
      clipQueue.filter((clip) => clip.damId === damId).map((clip) => clip.location)
    )
  ).sort();
  locations.forEach((location) => {
    const option = document.createElement('option');
    option.value = location;
    option.textContent = location;
    select.appendChild(option);
  });
};

const getScopedQueue = () => clipQueue.filter((clip) => clip.damId === state.damId);

const applyFilters = () => {
  let scoped = getScopedQueue();
  const { reviewType, startDate, endDate, location, sort } = state.filters;

  if (reviewType !== 'all') {
    scoped = scoped.filter((clip) => clip.reviewType === reviewType);
  }

  if (startDate) {
    const start = new Date(startDate);
    scoped = scoped.filter((clip) => new Date(clip.capturedAt) >= start);
  }

  if (endDate) {
    const end = new Date(endDate);
    scoped = scoped.filter((clip) => new Date(clip.capturedAt) <= end);
  }

  if (location) {
    scoped = scoped.filter((clip) => clip.location === location);
  }

  scoped.sort(sort === 'fifo' ? sortByDateAsc : sortByDateDesc);

  return scoped;
};

const renderMetrics = () => {
  const dam = damSummaries[state.damId];
  $('metric-total').textContent = formatNumber(dam.total);
  $('metric-manual').textContent = formatNumber(dam.manual);

  const pending = getScopedQueue().length;
  dam.pending = pending;
  $('metric-pending').textContent = formatNumber(pending);

  if (state.lastSync) {
    $('metric-sync').textContent = formatTimestamp(state.lastSync);
  } else {
    $('metric-sync').textContent = '--';
  }
};

const renderDamTable = () => {
  const tbody = $('dam-table');
  tbody.innerHTML = '';
  Object.entries(damSummaries)
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([dam, data]) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${dam}</td>
        <td>${formatNumber(data.total)}</td>
        <td>${formatNumber(data.manual)}</td>
        <td>${formatNumber(data.pending)}</td>
      `;
      if (dam === state.damId) {
        row.classList.add('active-dam');
      }
      tbody.appendChild(row);
    });
};

const renderChangeLog = () => {
  const list = $('change-log');
  list.innerHTML = '';
  if (state.changeLog.length === 0) {
    const empty = document.createElement('li');
    empty.textContent = 'No manual edits yet.';
    list.appendChild(empty);
    return;
  }

  state.changeLog.slice(0, 12).forEach((entry) => {
    const item = document.createElement('li');
    item.innerHTML = `
      <strong>${entry.title}</strong>
      <span>${entry.detail}</span>
    `;
    list.appendChild(item);
  });
};

const renderQueueIndicator = (total, index) => {
  const indicator = $('queue-indicator');
  indicator.innerHTML = '';
  if (total === 0) {
    indicator.textContent =
      'No clips require review. New detections will appear here as they fall below the confidence threshold.';
    return;
  }

  const position = document.createElement('span');
  position.innerHTML = `Clip <strong>${index + 1}</strong> of <strong>${total}</strong>`;

  const progress = document.createElement('div');
  progress.className = 'queue-progress';
  progress.style.setProperty('--progress', `${((index + 1) / total) * 100}%`);

  indicator.append(position, progress);
};

const createBoundingBox = (box) => {
  const element = document.createElement('div');
  element.className = `bounding-box${box.highlight ? ' highlight' : ''}`;
  element.style.left = `${box.x * 100}%`;
  element.style.top = `${box.y * 100}%`;
  element.style.width = `${box.width * 100}%`;
  element.style.height = `${box.height * 100}%`;

  const label = document.createElement('span');
  label.textContent = `${box.label} • ${box.confidence}%`;
  element.appendChild(label);

  return element;
};

const decorateRadioGroups = (form) => {
  form.querySelectorAll('.radio-pill').forEach((pill) => {
    const input = pill.querySelector('input');
    const setActive = () => {
      form
        .querySelectorAll(`input[name="${input.name}"]`)
        .forEach((radio) => radio.parentElement.classList.remove('active'));
      input.parentElement.classList.add('active');
    };
    if (input.checked) {
      pill.classList.add('active');
    }
    input.addEventListener('change', () => {
      setActive();
      if (input.name === 'action') {
        const countField = form.elements['count'];
        countField.disabled = input.value === 'dont_count';
        if (countField.disabled) {
          countField.value = 0;
        } else if (!countField.value) {
          countField.value = 1;
        }
      }
    });
  });
};

const buildCorrectionForm = (clip) => {
  const form = document.createElement('form');
  form.className = 'correction-form';
  form.innerHTML = `
    <div class="correction-form__row">
      <fieldset>
        <legend>Action</legend>
        <div class="radio-group">
          <label class="radio-pill active">
            <input type="radio" name="action" value="count" checked />
            <span>Count</span>
          </label>
          <label class="radio-pill">
            <input type="radio" name="action" value="dont_count" />
            <span>Don't count</span>
          </label>
        </div>
      </fieldset>
      <label>
        Count
        <input type="number" name="count" min="0" value="${clip.countGuess}" />
      </label>
    </div>
    <div class="correction-form__row">
      <label>
        Species
        <select name="species">
          ${speciesOptions
            .map(
              (species) =>
                `<option value="${species}"${species === clip.speciesGuess ? ' selected' : ''}>${species}</option>`
            )
            .join('')}
        </select>
      </label>
      <fieldset>
        <legend>Adipose presence</legend>
        <div class="radio-group">
          <label class="radio-pill ${clip.adiposeFlag === 'Present' ? 'active' : ''}">
            <input type="radio" name="adipose" value="Present" ${
              clip.adiposeFlag === 'Present' ? 'checked' : ''
            } />
            <span>Present</span>
          </label>
          <label class="radio-pill ${clip.adiposeFlag === 'Clipped' ? 'active' : ''}">
            <input type="radio" name="adipose" value="Clipped" ${
              clip.adiposeFlag === 'Clipped' ? 'checked' : ''
            } />
            <span>Clipped</span>
          </label>
        </div>
      </fieldset>
    </div>
    <label>
      Notes (optional)
      <textarea name="notes" rows="2" placeholder="Add context for this correction"></textarea>
    </label>
    <div class="action-bar">
      <button type="submit" class="ghost-button">Submit correction</button>
    </div>
  `;

  decorateRadioGroups(form);

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const action = data.get('action');
    const count = Number.parseInt(data.get('count'), 10) || 0;
    const payload = {
      action,
      count: action === 'dont_count' ? 0 : count,
      species: data.get('species'),
      adipose: data.get('adipose'),
      notes: data.get('notes')?.trim() ?? '',
      override: true,
    };
    finalizeClipReview(clip, payload);
  });

  return form;
};

const buildClipCard = (clip) => {
  const card = document.createElement('div');
  card.className = 'clip-card';

  const videoWrapper = document.createElement('div');
  videoWrapper.className = 'video-wrapper';
  const video = document.createElement('video');
  video.src = clip.clipUrl;
  video.autoplay = true;
  video.loop = true;
  video.controls = true;
  video.muted = true;

  const boxes = document.createElement('div');
  boxes.className = 'bounding-boxes';
  clip.boundingBoxes.forEach((box) => boxes.appendChild(createBoundingBox(box)));

  videoWrapper.append(video, boxes);

  const metadata = document.createElement('dl');
  metadata.className = 'clip-metadata';
  metadata.innerHTML = `
    <div>
      <dt>Clip ID</dt>
      <dd>${clip.id}</dd>
    </div>
    <div>
      <dt>Detected</dt>
      <dd>${formatTimestamp(clip.capturedAt)}</dd>
    </div>
    <div>
      <dt>Location</dt>
      <dd>${clip.location}</dd>
    </div>
    <div>
      <dt>Model guess</dt>
      <dd>${clip.speciesGuess} • ${clip.adiposeFlag}</dd>
    </div>
    <div>
      <dt>Model count</dt>
      <dd>${clip.countGuess}</dd>
    </div>
    <div>
      <dt>Review needs</dt>
      <dd>${clip.needs.join(', ')}</dd>
    </div>
    <div>
      <dt>Retention</dt>
      <dd>${formatRetentionCountdown(clip.capturedAt)}</dd>
    </div>
  `;

  const actionBar = document.createElement('div');
  actionBar.className = 'action-bar';
  const confirmButton = document.createElement('button');
  confirmButton.className = 'primary-button';
  confirmButton.type = 'button';
  confirmButton.textContent = 'Confirm model estimate';
  confirmButton.addEventListener('click', () =>
    finalizeClipReview(clip, {
      action: 'count',
      count: clip.countGuess,
      species: clip.speciesGuess,
      adipose: clip.adiposeFlag,
      notes: '',
      override: false,
    })
  );

  const helper = document.createElement('p');
  helper.className = 'helper-text';
  helper.textContent = 'One-click confirmation logs the species, count, and adipose flag as predicted.';

  actionBar.append(confirmButton, helper);

  card.append(videoWrapper, metadata, actionBar, buildCorrectionForm(clip));
  return card;
};

const renderClipStage = () => {
  const stage = $('clip-stage');
  stage.innerHTML = '';
  const queue = applyFilters();

  if (queue.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.innerHTML = `
      <h3>All caught up! 🎉</h3>
      <p>No clips require attention. Filters may be hiding reviewed items.</p>
    `;
    stage.appendChild(empty);
    renderQueueIndicator(0, 0);
    return;
  }

  if (state.queueIndex >= queue.length) {
    state.queueIndex = queue.length - 1;
  }

  const clip = queue[state.queueIndex];
  stage.appendChild(buildClipCard(clip));
  renderQueueIndicator(queue.length, state.queueIndex);
};

const updateUi = () => {
  renderMetrics();
  renderClipStage();
  renderDamTable();
  renderChangeLog();
};

const showToast = (message) => {
  const toast = $('toast');
  toast.textContent = message;
  toast.hidden = false;
  setTimeout(() => {
    toast.hidden = true;
  }, 3200);
};

const appendChangeLog = (clip, payload) => {
  const reviewedAt = new Date();
  const detailParts = [
    `${payload.action === 'dont_count' ? 'Marked as do not count' : `${payload.count} counted`}`,
    `${payload.species}`,
    `${payload.adipose}`,
  ];
  if (payload.notes) {
    detailParts.push(`Notes: ${payload.notes}`);
  }
  state.changeLog.unshift({
    title: `${clip.id} updated by ${state.user.name}`,
    detail: `${detailParts.join(' • ')} • ${formatTimestamp(reviewedAt)}`,
  });
};

const recordReviewHistory = (clip, payload) => {
  const entry = {
    ...clip,
    reviewer: state.user.name,
    reviewerEmail: state.user.email,
    reviewedAt: new Date().toISOString(),
    result: payload,
    finalSpecies: payload.species,
  };
  state.reviewHistory.unshift(entry);
};

const finalizeClipReview = (clip, payload) => {
  const queueIndex = clipQueue.findIndex((item) => item.id === clip.id);
  if (queueIndex === -1) return;

  clipQueue.splice(queueIndex, 1);
  recordReviewHistory(clip, payload);
  appendChangeLog(clip, payload);

  const dam = damSummaries[state.damId];
  dam.manual += 1;
  if (payload.action === 'count') {
    dam.total += payload.count;
  }

  state.lastSync = new Date();
  showToast('Clip synced. Images queued to improve the training dataset.');
  state.queueIndex = Math.max(0, state.queueIndex - 1);
  updateUi();
};

const handleSearch = (event) => {
  const query = event.target.value.trim().toLowerCase();
  const resultsContainer = $('search-results');
  if (!query) {
    resultsContainer.classList.add('hidden');
    resultsContainer.innerHTML = '';
    return;
  }

  const matches = [...getScopedQueue(), ...state.reviewHistory]
    .filter((clip) =>
      [clip.id, clip.finalSpecies, clip.speciesGuess, clip.result?.species, clip.location, clip.reviewer]
        .filter(Boolean)
        .some((value) => value.toLowerCase().includes(query))
    )
    .slice(0, 6);

  if (matches.length === 0) {
    resultsContainer.innerHTML = '<p>No clips found.</p>';
  } else {
    const list = document.createElement('ul');
    matches.forEach((clip) => {
      const item = document.createElement('li');
      item.innerHTML = `
        <strong>${clip.id}</strong> – ${(clip.finalSpecies ?? clip.speciesGuess ?? clip.result?.species) ?? 'Unknown'} (${formatTimestamp(
        clip.reviewedAt ?? clip.capturedAt
      )})
      `;
      item.addEventListener('click', () => {
        const queue = applyFilters();
        const index = queue.findIndex((entry) => entry.id === clip.id);
        if (index >= 0) {
          state.queueIndex = index;
          renderClipStage();
        }
        resultsContainer.classList.add('hidden');
      });
      list.appendChild(item);
    });
    resultsContainer.innerHTML = '';
    resultsContainer.appendChild(list);
  }

  resultsContainer.classList.remove('hidden');
};

const downloadReviewed = () => {
  if (state.reviewHistory.length === 0) {
    showToast('No reviewed clips available yet.');
    return;
  }

  const header = [
    'clip_id',
    'dam',
    'location',
    'captured_at',
    'reviewed_at',
    'action',
    'count',
    'species',
    'adipose',
    'reviewer',
    'notes',
  ];

  const rows = state.reviewHistory.map((entry) => [
    entry.id,
    entry.damId,
    entry.location,
    entry.capturedAt,
    entry.reviewedAt,
    entry.result.action,
    entry.result.count,
    entry.result.species,
    entry.result.adipose,
    entry.reviewer,
    entry.result.notes ?? '',
  ]);

  const csv = [header, ...rows]
    .map((line) => line.map((value) => `"${String(value ?? '').replace(/"/g, '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `arkcv-reviewed-clips-${Date.now()}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

const attachFilterHandlers = () => {
  document.querySelectorAll('[data-sort]').forEach((button) => {
    button.addEventListener('click', () => {
      state.filters.sort = button.dataset.sort;
      document.querySelectorAll('[data-sort]').forEach((btn) => btn.classList.remove('active'));
      button.classList.add('active');
      state.queueIndex = 0;
      updateUi();
    });
  });

  document.querySelectorAll('[data-type]').forEach((button) => {
    button.addEventListener('click', () => {
      state.filters.reviewType = button.dataset.type;
      document.querySelectorAll('[data-type]').forEach((btn) => btn.classList.remove('active'));
      button.classList.add('active');
      state.queueIndex = 0;
      updateUi();
    });
  });

  $('filter-start').addEventListener('change', (event) => {
    state.filters.startDate = event.target.value;
    state.queueIndex = 0;
    updateUi();
  });

  $('filter-end').addEventListener('change', (event) => {
    state.filters.endDate = event.target.value;
    state.queueIndex = 0;
    updateUi();
  });

  $('filter-location').addEventListener('change', (event) => {
    state.filters.location = event.target.value;
    state.queueIndex = 0;
    updateUi();
  });
};

const attachQueueNavHandlers = () => {
  $('queue-prev').addEventListener('click', () => {
    state.queueIndex = Math.max(0, state.queueIndex - 1);
    renderClipStage();
  });

  $('queue-next').addEventListener('click', () => {
    const total = applyFilters().length;
    state.queueIndex = Math.min(total - 1, state.queueIndex + 1);
    renderClipStage();
  });
};

const attachGlobalHandlers = () => {
  $('global-search').addEventListener('input', handleSearch);
  $('download-reviewed').addEventListener('click', downloadReviewed);
  document.addEventListener('click', (event) => {
    const results = $('search-results');
    if (!results.classList.contains('hidden') && !results.contains(event.target) && event.target !== $('global-search')) {
      results.classList.add('hidden');
    }
  });
};

const handleLogout = () => {
  if (state.auth0Client && !state.isDemo) {
    state.auth0Client.logout({ logoutParams: { returnTo: window.location.origin } });
  }
  $('app-shell').classList.add('hidden');
  $('auth-screen').classList.remove('hidden');
  state.user = null;
  state.damId = null;
  state.isDemo = false;
};

const enterConsole = (user, isDemo = false) => {
  state.user = user;
  state.damId = mapUserToDam(user);
  state.isDemo = isDemo;

  if (!state.damId) {
    showAuthError('No dam assignment found for this account.');
    state.user = null;
    state.isDemo = false;
    return;
  }

  hideAuthError();
  $('auth-screen').classList.add('hidden');
  $('app-shell').classList.remove('hidden');

  $('user-name').textContent = user.name;
  $('user-role').textContent = user.role ?? 'Operator';
  $('dam-pill').textContent = state.damId;

  state.filters = { sort: 'fifo', reviewType: 'all', startDate: '', endDate: '', location: '' };
  state.reviewHistory = [];
  state.changeLog = [];
  state.lastSync = null;
  resetClipQueue();
  document.querySelectorAll('[data-sort]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.sort === state.filters.sort);
  });
  document.querySelectorAll('[data-type]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.type === state.filters.reviewType);
  });
  $('filter-start').value = '';
  $('filter-end').value = '';
  $('filter-location').value = '';

  populateLocationFilter(state.damId);
  state.queueIndex = 0;
  updateUi();
};

const setupAuth0 = async () => {
  if (!window.createAuth0Client) {
    showAuthError('Auth0 SDK did not load. Check your network connection.');
    return;
  }

  if (!isAuthConfigured()) {
    showAuthError('Update authConfig with your Auth0 domain and client ID.');
    $('auth-login').addEventListener('click', () =>
      showAuthError('Auth0 is not configured. Edit ui/app.js with your credentials.')
    );
    return;
  }

  try {
    state.auth0Client = await window.createAuth0Client({
      domain: authConfig.domain,
      clientId: authConfig.clientId,
      cacheLocation: 'localstorage',
      useRefreshTokens: true,
      authorizationParams: {
        audience: authConfig.audience,
        redirect_uri: window.location.origin,
      },
    });

    const query = window.location.search;
    if (query.includes('code=') && query.includes('state=')) {
      await state.auth0Client.handleRedirectCallback();
      window.history.replaceState({}, document.title, window.location.pathname);
    }

    const isAuthenticated = await state.auth0Client.isAuthenticated();

    if (isAuthenticated) {
      const userProfile = await state.auth0Client.getUser();
      const mappedUser = {
        id: userProfile.sub,
        name: userProfile.name ?? userProfile.email ?? 'Auth0 User',
        email: userProfile.email,
        role: userProfile['https://ark.cv/role'] ?? 'Operator',
        isDemo: false,
      };
      enterConsole(mappedUser, false);
    }

    $('auth-login').addEventListener('click', async () => {
      await state.auth0Client.loginWithRedirect({
        authorizationParams: {
          redirect_uri: window.location.origin,
        },
      });
    });
  } catch (error) {
    console.error('Auth0 initialization failed', error);
    showAuthError('Unable to initialize Auth0. Check credentials and redirect URIs.');
  }
};

const init = async () => {
  attachFilterHandlers();
  attachQueueNavHandlers();
  attachGlobalHandlers();
  $('logout-button').addEventListener('click', handleLogout);
  $('auth-demo').addEventListener('click', () => enterConsole(demoAccount, true));

  await setupAuth0();
};

document.addEventListener('DOMContentLoaded', init);
