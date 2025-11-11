const stats = {
  ytd: 12842,
  mtd: 982,
  wtd: 134,
};

const trendData = [
  { day: 'Mon', value: 18 },
  { day: 'Tue', value: 22 },
  { day: 'Wed', value: 15 },
  { day: 'Thu', value: 28 },
  { day: 'Fri', value: 32 },
  { day: 'Sat', value: 24 },
  { day: 'Sun', value: 19 },
];

const lowConfidenceDetections = [
  {
    id: 'LC-2301',
    species: 'Chinook Salmon',
    confidence: 0.42,
    detectedAt: '2024-03-28T07:23:00Z',
    location: 'Station 12B',
    adipose: true,
    imageUrl: 'https://images.unsplash.com/photo-1585128792020-803d29415281?auto=format&fit=crop&w=600&q=80',
  },
  {
    id: 'LC-2294',
    species: 'Steelhead Trout',
    confidence: 0.37,
    detectedAt: '2024-03-27T15:41:00Z',
    location: 'Station 07A',
    adipose: false,
    imageUrl: 'https://images.unsplash.com/photo-1508184964240-ee94ad12880e?auto=format&fit=crop&w=600&q=80',
  },
  {
    id: 'LC-2289',
    species: 'Sockeye Salmon',
    confidence: 0.35,
    detectedAt: '2024-03-27T04:18:00Z',
    location: 'Station 04C',
    adipose: true,
    imageUrl: 'https://images.unsplash.com/photo-1533090161767-e6ffed986c88?auto=format&fit=crop&w=600&q=80',
  },
];

const state = {
  detections: structuredClone(lowConfidenceDetections),
  filters: { species: '', adipose: '', timeframe: '7d' },
  ui: { menuOpen: false, modalOpen: false, currentChangeId: null },
};

const SPECIES_OPTIONS = [
  'Chinook', 'Coho', 'Lamprey', 'Sockeye', 'Steelhead', 'Bull Trout'
];

const formatNumber = (value) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);

const formatConfidence = (value) => `${Math.round(value * 100)}%`;

const formatTimestamp = (isoString) =>
  new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(isoString));

const renderStats = () => {
  document.getElementById('ytd-count').textContent = formatNumber(stats.ytd);
  document.getElementById('mtd-count').textContent = formatNumber(stats.mtd);
  document.getElementById('wtd-count').textContent = formatNumber(stats.wtd);
};

const renderTrend = () => {
  const container = document.getElementById('trend-chart');
  container.innerHTML = '';

  const maxValue = Math.max(...trendData.map((entry) => entry.value));

  trendData.forEach((entry) => {
    const bar = document.createElement('div');
    bar.className = 'trend-bar';

    const valueLabel = document.createElement('span');
    valueLabel.className = 'value';
    valueLabel.textContent = entry.value;

    const barShape = document.createElement('div');
    barShape.className = 'bar';
    barShape.style.height = `${(entry.value / maxValue) * 120}px`;

    const dayLabel = document.createElement('span');
    dayLabel.textContent = entry.day;

    bar.append(valueLabel, barShape, dayLabel);
    container.appendChild(bar);
  });
};

const handleAccept = (id) => {
  state.detections = state.detections.filter((detection) => detection.id !== id);
  renderDetections();
};

// Modal-based change species (using dropdown)
const openChangeSpeciesModal = (id) => {
  state.ui.currentChangeId = id;
  const detection = state.detections.find((d) => d.id === id);
  if (!detection) return;
  const select = document.getElementById('modal-species-select');
  // Try to preselect based on the leading word (e.g., "Chinook Salmon" -> "Chinook")
  const lead = (detection.species || '').split(' ')[0];
  select.value = SPECIES_OPTIONS.includes(lead) ? lead : '';
  document.getElementById('modal-overlay').classList.add('show');
  state.ui.modalOpen = true;
};

const closeChangeSpeciesModal = () => {
  document.getElementById('modal-overlay').classList.remove('show');
  state.ui.modalOpen = false;
  state.ui.currentChangeId = null;
};

const saveChangeSpeciesModal = () => {
  const id = state.ui.currentChangeId;
  const detection = state.detections.find((d) => d.id === id);
  if (!detection) return;
  const value = document.getElementById('modal-species-select').value.trim();
  if (value) {
    detection.species = value;
    detection.confidence = 1;
    renderDetections();
  }
  closeChangeSpeciesModal();
};

const getFilteredDetections = () => {
  const { species, adipose, timeframe } = state.filters;
  const now = new Date('2024-03-29T00:00:00Z'); // demo reference date
  const minDate = (() => {
    if (timeframe === '7d') return new Date(now.getTime() - 7 * 86400000);
    if (timeframe === '30d') return new Date(now.getTime() - 30 * 86400000);
    return new Date(now.getFullYear(), 0, 1);
  })();
  return state.detections.filter((d) => {
    if (species && d.species !== species) return false;
    if (adipose !== '') {
      const want = adipose === 'true';
      if (!!d.adipose !== want) return false;
    }
    if (d.detectedAt) {
      const t = new Date(d.detectedAt);
      if (t < minDate || t > now) return false;
    }
    return true;
  });
};

const renderDetections = () => {
  const container = document.getElementById('review-container');
  const template = document.getElementById('review-card-template');
  container.innerHTML = '';

  const list = getFilteredDetections();

  if (list.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <h3>All caught up! \uD83C\uDF89</h3>
      <p>There are no low-confidence detections waiting for review.</p>
    `;
    container.appendChild(emptyState);
    return;
  }

  list.forEach((detection) => {
    const card = template.content.firstElementChild.cloneNode(true);
    const image = card.querySelector('.review-image');
    const speciesName = card.querySelector('.species-name');
    const confidence = card.querySelector('.confidence');
    const timestamp = card.querySelector('.timestamp');
    const location = card.querySelector('.location');
    const acceptButton = card.querySelector('.action-button.accept');
    const changeButton = card.querySelector('.action-button.change');

    image.src = detection.imageUrl;
    image.alt = `${detection.species} detection`;
    speciesName.textContent = `${detection.species} (${detection.id})`;
    confidence.textContent = formatConfidence(detection.confidence);
    timestamp.textContent = formatTimestamp(detection.detectedAt);
    location.textContent = detection.location;

    acceptButton.addEventListener('click', () => handleAccept(detection.id));
    changeButton.addEventListener('click', () => openChangeSpeciesModal(detection.id));

    container.appendChild(card);
  });
};

const renderFilters = () => {
  const select = document.getElementById('filter-species');
  // Ensure fixed species list (already present in HTML). Nothing dynamic needed.
  // No-op placeholder for parity with init flow.
};

const wireUI = () => {
  // Profile menu
  const avatarButton = document.getElementById('avatarButton');
  const menu = document.getElementById('profileMenu');
  const logoutButton = document.getElementById('logoutButton');
  avatarButton.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = menu.classList.toggle('open');
    avatarButton.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  document.addEventListener('click', () => menu.classList.remove('open'));
  logoutButton.addEventListener('click', () => alert('Logout (placeholder)'));

  // Filters
  document.getElementById('filter-species').addEventListener('change', (e) => {
    state.filters.species = e.target.value;
    renderDetections();
  });
  document.getElementById('filter-adipose').addEventListener('change', (e) => {
    state.filters.adipose = e.target.value;
    renderDetections();
  });
  document.getElementById('filter-timeframe').addEventListener('change', (e) => {
    state.filters.timeframe = e.target.value;
    renderDetections();
  });

  // Modal events
  document.getElementById('modal-cancel').addEventListener('click', closeChangeSpeciesModal);
  document.getElementById('modal-save').addEventListener('click', saveChangeSpeciesModal);
  document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') closeChangeSpeciesModal();
  });
};

const init = () => {
  renderStats();
  renderTrend();
  renderFilters();
  renderDetections();
  wireUI();
};

document.addEventListener('DOMContentLoaded', init);
