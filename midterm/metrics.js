// Metrics and evaluation rendering

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  function confusionMatrix(pred, truth, numClasses) {
    const m = Array.from({length:numClasses}, () => Array(numClasses).fill(0));
    for (let i=0;i<truth.length;i++) m[truth[i]][pred[i]] += 1;
    return m;
  }

  function renderEvaluation(targetEl, metrics) {
    const { loss, accuracy, confMat } = metrics;
    const classes = NS.CLASSES;
    targetEl.innerHTML = `
      <div class="metrics">
        <div><strong>Loss</strong>: ${loss.toFixed(4)}</div>
        <div><strong>Accuracy</strong>: ${(accuracy*100).toFixed(2)}%</div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th></th>${classes.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
          <tbody>
            ${confMat.map((row,i)=>`<tr><th>${classes[i]}</th>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  // Lightweight chart renderer for training curves (loss and accuracy)
  function renderTrainingChart(canvas, history) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    const epochs = history.epoch || Array.from({length: (history.loss||[]).length}, (_,i)=>i+1);
    const series = [];
    if (history.loss && history.loss.length) series.push({ key: 'loss', data: history.loss, color: '#f59e0b' });
    if (history.val_loss && history.val_loss.length) series.push({ key: 'val_loss', data: history.val_loss, color: '#f97316' });
    if (history.acc && history.acc.length) series.push({ key: 'acc', data: history.acc, color: '#22c55e' });
    if (history.val_acc && history.val_acc.length) series.push({ key: 'val_acc', data: history.val_acc, color: '#16a34a' });
    if (history.accuracy && history.accuracy.length) series.push({ key: 'accuracy', data: history.accuracy, color: '#22c55e' });
    if (history.val_accuracy && history.val_accuracy.length) series.push({ key: 'val_accuracy', data: history.val_accuracy, color: '#16a34a' });

    if (!series.length) return;

    const pad = { l: 36, r: 8, t: 8, b: 20 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const n = Math.max(1, epochs.length);
    const xFor = (i) => pad.l + (i/(n-1)) * plotW;

    let yMin = Infinity, yMax = -Infinity;
    for (const s of series) for (const v of s.data) { if (v<yMin) yMin=v; if (v>yMax) yMax=v; }
    if (!isFinite(yMin) || !isFinite(yMax)) return;
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yFor = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;

    // Axes
    ctx.strokeStyle = '#223047'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + plotH); ctx.lineTo(pad.l + plotW, pad.t + plotH);
    ctx.stroke();
    ctx.fillStyle = '#9da7b3'; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let t=0;t<=4;t++) {
      const v = yMin + (t/4)*(yMax - yMin);
      const y = yFor(v);
      ctx.fillText(v.toFixed(3), pad.l - 6, y);
      ctx.strokeStyle = '#223047'; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+plotW, y); ctx.stroke();
    }

    // Lines
    ctx.lineWidth = 2;
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      for (let i=0;i<s.data.length;i++) {
        const x = xFor(i);
        const y = yFor(s.data[i]);
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }

    // Legend
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    let lx = pad.l + 6, ly = pad.t + 6;
    for (const s of series) {
      ctx.fillStyle = s.color; ctx.fillRect(lx, ly+2, 10, 10);
      ctx.fillStyle = '#e6edf3'; ctx.fillText(s.key, lx + 14, ly);
      lx += 70;
    }
  }

  function renderTrainingSummary(targetEl, history){
    if (!targetEl) return;
    const last = (arr) => (arr && arr.length ? arr[arr.length-1] : undefined);
    const loss = last(history.loss);
    const acc = last(history.acc || history.accuracy);
    const val_loss = last(history.val_loss);
    const val_acc = last(history.val_acc || history.val_accuracy);
    targetEl.innerHTML = `
      <div class="metrics">
        ${loss!=null?`<div><strong>Final loss</strong>: ${loss.toFixed(4)}</div>`:''}
        ${acc!=null?`<div><strong>Final acc</strong>: ${(acc*100).toFixed(2)}%</div>`:''}
        ${val_loss!=null?`<div><strong>Val loss</strong>: ${val_loss.toFixed(4)}</div>`:''}
        ${val_acc!=null?`<div><strong>Val acc</strong>: ${(val_acc*100).toFixed(2)}%</div>`:''}
      </div>
    `;
  }

  NS.confusionMatrix = confusionMatrix;
  NS.renderEvaluation = renderEvaluation;
  NS.renderTrainingChart = renderTrainingChart;
  NS.renderTrainingSummary = renderTrainingSummary;
})();


