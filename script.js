const toggleBtn = document.getElementById('toggleBtn');
const candle = document.getElementById('candle');
const flame = document.getElementById('flame');

let isOn = true;

function updateCandle() {
  document.body.classList.toggle('off', !isOn);
  toggleBtn.textContent = isOn ? 'Turn Off' : 'Turn On';
  candle.setAttribute('aria-pressed', String(isOn));
  flame.setAttribute('aria-hidden', String(!isOn));
}

toggleBtn.addEventListener('click', () => {
  isOn = !isOn;
  updateCandle();
});

updateCandle();
