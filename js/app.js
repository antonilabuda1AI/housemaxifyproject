// House Maxify prototype JS
// Small enhancements only; the app works without JS.

document.addEventListener('DOMContentLoaded', () => {
  // Animate result progress bar if present
  const bar = document.querySelector('.progress__bar');
  if (bar) {
    requestAnimationFrame(() => {
      // Decorative width; not tied to any metric
      bar.style.width = '72%';
    });
  }

  // Basic UX: prevent invalid characters in number inputs (e/E/+/- for most)
  document.querySelectorAll('input[type="number"]').forEach((el) => {
    el.addEventListener('keydown', (e) => {
      const blocked = ['e', 'E'];
      if (blocked.includes(e.key)) e.preventDefault();
    });
    el.addEventListener('blur', () => {
      if (el.min !== '' && +el.value < +el.min) el.value = el.min;
      if (el.max !== '' && +el.value > +el.max) el.value = el.max;
    });
  });
});
