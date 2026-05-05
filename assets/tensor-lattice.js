(() => {
  const canvas = document.querySelector("#tensor-canvas");
  if (!canvas) return;

  const context = canvas.getContext("2d");
  const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const palette = ["#008f8c", "#4f9d69", "#e85d4f", "#c58a00"];
  let width = 0;
  let height = 0;
  let nodes = [];
  let rafId = null;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    width = rect.width;
    height = rect.height;
    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.max(1, Math.floor(height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    buildNodes();
  }

  function buildNodes() {
    const centerX = width * 0.68;
    const centerY = height * 0.47;
    const depth = Math.min(width, height) * 0.12;
    const gap = Math.min(width, height) * 0.09;
    nodes = [];

    for (let z = -2; z <= 2; z += 1) {
      for (let y = -2; y <= 2; y += 1) {
        for (let x = -2; x <= 2; x += 1) {
          const projectionX = centerX + x * gap + z * depth;
          const projectionY = centerY + y * gap - z * depth * 0.54;
          const active = (x + y + z + 6) % 4 === 0 || (x === 0 && y === 0);
          nodes.push({
            x: projectionX,
            y: projectionY,
            z,
            active,
            color: palette[Math.abs(x + y * 2 + z * 3) % palette.length],
            phase: Math.random() * Math.PI * 2,
          });
        }
      }
    }
  }

  function draw(time = 0) {
    context.clearRect(0, 0, width, height);
    context.lineWidth = 1.25;

    for (const a of nodes) {
      for (const b of nodes) {
        const horizontal = Math.abs(a.x - b.x) < 2 && Math.abs(a.y - b.y) < 120;
        const diagonal = Math.abs(a.x - b.x) < 95 && Math.abs(a.y - b.y) < 55 && a.z !== b.z;
        if (a === b || (!horizontal && !diagonal)) continue;

        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (distance > 115) continue;

        context.beginPath();
        context.strokeStyle = a.active && b.active ? "rgba(0, 143, 140, 0.24)" : "rgba(23, 32, 42, 0.07)";
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.stroke();
      }
    }

    for (const node of nodes) {
      const pulse = mediaQuery.matches ? 0 : Math.sin(time * 0.0014 + node.phase) * 1.5;
      const radius = node.active ? 4.8 + pulse : 2.4;
      context.beginPath();
      context.fillStyle = node.active ? node.color : "rgba(23, 32, 42, 0.28)";
      context.arc(node.x, node.y, radius, 0, Math.PI * 2);
      context.fill();
    }

    if (!mediaQuery.matches) {
      rafId = window.requestAnimationFrame(draw);
    }
  }

  function start() {
    if (rafId) window.cancelAnimationFrame(rafId);
    rafId = null;
    draw();
  }

  window.addEventListener("resize", () => {
    resize();
    start();
  });
  mediaQuery.addEventListener("change", start);
  resize();
  start();
})();
