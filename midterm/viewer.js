// Orbit viewer (split from utils)

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  function createOrbitViewer(canvas) {
    let yaw = 0.6, pitch = 0.5;
    // Auto-fit scale derived from data; userZoom is multiplicative for manual zoom
    let autoRadius = 1, userZoom = 1;
    let isDragging = false, lastX = 0, lastY = 0;
    let currentCloud = null;

    function projectPoint([x,y,z]) {
      const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
      const xr = x * cosY - y * sinY;
      const yr = x * sinY + y * cosY;
      const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
      const yr2 = yr * cosP - z * sinP;
      const zr2 = yr * sinP + z * cosP;
      const totalRadius = Math.max(1e-6, autoRadius * userZoom);
      const s = Math.min(canvas.width, canvas.height) * (0.45 / totalRadius);
      const cx = canvas.width/2, cy = canvas.height/2;
      return { px: cx + xr * s, py: cy - yr2 * s, depth: zr2 };
    }

    function draw(cloud, color="#e6edf3") {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      const axes = [[1,0,0],[0,1,0],[0,0,1]];
      const colors = ["#ef4444aa","#22c55eaa","#3b82f6aa"]; // x,y,z
      for (let i=0;i<3;i++) {
        const a = projectPoint([0,0,0]); const b = projectPoint(axes[i]);
        ctx.strokeStyle = colors[i]; ctx.beginPath(); ctx.moveTo(a.px,a.py); ctx.lineTo(b.px,b.py); ctx.stroke();
      }
      const projected = cloud.map(p => ({...projectPoint(p)}));
      const order = projected.map((p,i)=>[p.depth,i]).sort((a,b)=>a[0]-b[0]).map(([,i])=>i);
      ctx.fillStyle = color;
      for (const i of order) { const p = projected[i]; const size = Math.max(1, 2 + p.depth * 1.2); ctx.fillRect(p.px, p.py, size, size); }
    }

    function onDown(e){ isDragging = true; lastX=e.clientX; lastY=e.clientY; }
    function onUp(){ isDragging = false; }
    function onMove(e){ if(!isDragging) return; const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY; yaw+=dx*0.01; pitch+=dy*0.01; pitch=Math.max(-Math.PI/2+0.01, Math.min(Math.PI/2-0.01, pitch)); if(currentCloud) draw(currentCloud); }
    function onWheel(e){
      e.preventDefault();
      const factor = Math.exp(-e.deltaY*0.001);
      // Wide bounds so user can meaningfully zoom while keeping initial auto-fit
      userZoom = Math.max(0.05, Math.min(50, userZoom * factor));
      if (currentCloud) draw(currentCloud);
    }

    function computeAutoRadius(cloud){
      let r = 0;
      for (let i=0;i<cloud.length;i++) {
        const p = cloud[i];
        const rr = Math.hypot(p[0], p[1], p[2]);
        if (rr > r) r = rr;
      }
      return Math.max(1e-6, r);
    }

    function render(cloud, color){
      currentCloud = cloud;
      autoRadius = computeAutoRadius(cloud);
      // Keep userZoom as-is so zoom persists across renders; ensure draw fits by autoRadius
      draw(cloud, color);
    }

    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    return {
      render,
      setYaw:(v)=>{ yaw=v; if(currentCloud) draw(currentCloud); },
      setPitch:(v)=>{ pitch=v; if(currentCloud) draw(currentCloud); },
      // Back-compat: interpret setRadius as setting overall radius; map to userZoom
      setRadius:(v)=>{ userZoom = Math.max(1e-6, v) / Math.max(1e-6, autoRadius); if(currentCloud) draw(currentCloud); }
    };
  }

  NS.createOrbitViewer = createOrbitViewer;
})();


