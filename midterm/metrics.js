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

  NS.confusionMatrix = confusionMatrix;
  NS.renderEvaluation = renderEvaluation;
})();


