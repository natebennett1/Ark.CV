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
    imageUrl: 'https://images.unsplash.com/photo-1585128792020-803d29415281?auto=format&fit=crop&w=600&q=80',
  },
  {
    id: 'LC-2294',
    species: 'Steelhead Trout',
    confidence: 0.37,
    detectedAt: '2024-03-27T15:41:00Z',
    location: 'Station 07A',
    imageUrl: 'https://images.unsplash.com/photo-1508184964240-ee94ad12880e?auto=format&fit=crop&w=600&q=80',
  },
  {
    id: 'LC-2289',
    species: 'Sockeye Salmon',
    confidence: 0.35,
    detectedAt: '2024-03-27T04:18:00Z',
    location: 'Station 04C',
    imageUrl: 'https://images.unsplash.com/photo-1533090161767-e6ffed986c88?auto=format&fit=crop&w=600&q=80',
  },
];

const state = {
  detections: structuredClone(lowConfidenceDetections),
};

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

const handleChangeSpecies = (id) => {
  const detection = state.detections.find((item) => item.id === id);
  if (!detection) return;

  const newSpecies = prompt('Enter the correct species name:', detection.species);

  if (newSpecies && newSpecies.trim().length > 0) {
    detection.species = newSpecies.trim();
    detection.confidence = 1;
    renderDetections();
  }
};

const renderDetections = () => {
  const container = document.getElementById('review-container');
  const template = document.getElementById('review-card-template');
  container.innerHTML = '';

  if (state.detections.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <h3>All caught up! \uD83C\uDF89</h3>
      <p>There are no low-confidence detections waiting for review.</p>
    `;
    container.appendChild(emptyState);
    return;
  }

  state.detections.forEach((detection) => {
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
    changeButton.addEventListener('click', () => handleChangeSpecies(detection.id));

    container.appendChild(card);
  });
};

const init = () => {
  renderStats();
  renderTrend();
  renderDetections();
};

document.addEventListener('DOMContentLoaded', init);
