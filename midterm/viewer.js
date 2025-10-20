// Orbit viewer (split from utils)

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  function createOrbitViewer(canvas) {
    let yaw = 0.6, pitch = 0.5, radius = 2.2;
    let isDragging = false, lastX = 0, lastY = 0;
    let currentCloud = null;

    function projectPoint([x,y,z]) {
      const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
      const xr = x * cosY - y * sinY;
      const yr = x * sinY + y * cosY;
      const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
      const yr2 = yr * cosP - z * sinP;
      const zr2 = yr * sinP + z * cosP;
      const s = Math.min(canvas.width, canvas.height) * (0.45 / radius);
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
    function onWheel(e){ e.preventDefault(); const factor=Math.exp(-e.deltaY*0.001); radius=Math.max(0.5, Math.min(5, radius*factor)); if(currentCloud) draw(currentCloud); }

    function render(cloud, color){ currentCloud = cloud; draw(cloud, color); }

    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    return { render, setYaw:(v)=>{yaw=v; if(currentCloud)draw(currentCloud);}, setPitch:(v)=>{pitch=v; if(currentCloud)draw(currentCloud);}, setRadius:(v)=>{radius=v; if(currentCloud)draw(currentCloud);} };
  }

  NS.createOrbitViewer = createOrbitViewer;
})();


