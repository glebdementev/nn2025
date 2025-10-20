// Wire UI, dataset pipeline, training and evaluation

(() => {
  const U = window.PointCloudUtils;
  const M = window.PointNetTiny;

  const uploadInput = document.getElementById('upload-csv');
  const generateBtn = document.getElementById('generate-dataset');
  const exportBtn = document.getElementById('export-dataset');
  const trainBtn = document.getElementById('train-model');
  const evalBtn = document.getElementById('evaluate-model');
  const classSelect = document.getElementById('class-select');
  const fillToggle = document.getElementById('fill-toggle');
  const noiseSlider = document.getElementById('noise-slider');
  const jitterSlider = document.getElementById('jitter-slider');
  const pointsSlider = document.getElementById('points-slider');
  const samplesSlider = document.getElementById('samples-slider');
  const ratioMinSlider = document.getElementById('ratio-min-slider');
  const ratioMaxSlider = document.getElementById('ratio-max-slider');
  const noiseValue = document.getElementById('noise-value');
  const jitterValue = document.getElementById('jitter-value');
  const pointsValue = document.getElementById('points-value');
  const samplesValue = document.getElementById('samples-value');
  const ratioMinValue = document.getElementById('ratio-min-value');
  const ratioMaxValue = document.getElementById('ratio-max-value');
  const canvas = document.getElementById('point-cloud-canvas');
  const prevBtn = document.getElementById('prev-sample');
  const nextBtn = document.getElementById('next-sample');
  const viewModeSel = document.getElementById('view-mode');
  const sampleInfo = document.getElementById('sample-info');
  const legendEl = document.getElementById('legend');
  const evalDiv = document.getElementById('evaluation-results');
  const logDiv = document.getElementById('training-log');

  let noise = parseFloat(noiseSlider.value);
  let jitter = parseFloat(jitterSlider.value);
  let pointsPerCloud = parseInt(pointsSlider.value, 10);
  let samplesPerClass = parseInt(samplesSlider.value, 10);
  let ratioMin = Math.max(1, Math.min(6, parseFloat(ratioMinSlider.value)));
  let ratioMax = Math.max(1, Math.min(6, parseFloat(ratioMaxSlider.value)));
  if (ratioMin > ratioMax) { const tmp = ratioMin; ratioMin = ratioMax; ratioMax = tmp; }
  let fill = !!(fillToggle && fillToggle.checked);

  let dataset = { data: [], labels: [] };
  let datasetIndex = 0;
  let evalCache = null; // { data, labels, preds, conf, loss, accuracy }
  let evalIndex = 0;
  let trainTensors = null; // {inputs, labels, valInputs, valLabels}
  let model = null;

  const CLASS_COLOR = {
    pyramid: '#f97316',
    box: '#3b82f6',
    cylinder: '#a855f7',
    ellipsoid: '#10b981',
    paraboloid: '#eab308',
    cone: '#ef4444',
  };

  function logLine(text) {
    logDiv.textContent += (text + '\n');
    logDiv.scrollTop = logDiv.scrollHeight;
  }

  function ensureSamePointCount(cloud, targetPoints) {
    // If generated has > target, sample; if < target, repeat
    const out = [];
    if (cloud.length === 0) return Array.from({length:targetPoints}, () => [0,0,0]);
    if (cloud.length >= targetPoints) {
      for (let i=0;i<targetPoints;i++) out.push(cloud[Math.floor(Math.random()*cloud.length)]);
    } else {
      for (let i=0;i<targetPoints;i++) out.push(cloud[i % cloud.length]);
    }
    return out;
  }

  async function handleUpload(ev) {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const cloud = await U.processCSVFile(file);
    const fixed = U.normalizeCloud ? U.normalizeCloud(cloud) : cloud; // fallback
    const resized = ensureSamePointCount(fixed, pointsPerCloud);
    viewer.render(resized);
    sampleInfo.textContent = 'Uploaded cloud';
  }

  function handleGenerate() {
    dataset = U.createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter, ratioMin, ratioMax, fill);
    datasetIndex = 0;
    renderCurrent();
  }

  function handleExport() {
    if (!dataset.data.length) return;
    U.exportDatasetCSV(dataset, U.CLASSES);
  }

  async function prepareTensors() {
    if (!dataset.data.length) {
      dataset = U.createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter, ratioMin, ratioMax, fill);
    }
    const { trainData, trainLabels, valData, valLabels } = U.splitTrainVal(dataset.data, dataset.labels, 0.2);
    const inputs = U.cloudsToTensor(trainData);
    const labels = U.labelsToOneHot(trainLabels, U.CLASSES.length);
    const valInputs = U.cloudsToTensor(valData);
    const valLbls = U.labelsToOneHot(valLabels, U.CLASSES.length);
    return { inputs, labels, valInputs, valLbls };
  }

  async function handleTrain() {
    logDiv.textContent = '';
    const { inputs, labels, valInputs, valLbls } = await prepareTensors();
    trainTensors = { inputs, labels, valInputs, valLbls };
    if (!model || model.inputs[0].shape[1] !== pointsPerCloud || model.outputs[0].shape[1] !== U.CLASSES.length) {
      model = M.buildPointNetTiny(pointsPerCloud, U.CLASSES.length);
    }
    logLine('Starting training...');
    await model.fit(inputs, labels, {
      epochs: 30,
      batchSize: 32,
      validationData: [valInputs, valLbls],
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          logLine(`Epoch ${epoch+1} | loss=${logs.loss.toFixed(4)} acc=${(logs.acc??logs.accuracy).toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} val_acc=${(logs.val_acc??logs.val_accuracy).toFixed(4)}`);
        }
      }
    });
    logLine('Training complete.');
  }

  function argMaxPerRow(t) {
    return Array.from(t.argMax(-1).dataSync());
  }

  async function handleEvaluate() {
    if (!model) return;
    // Create a fresh eval set to avoid contamination
    const evalSet = U.createToyDataset(Math.max(60, Math.floor(samplesPerClass*0.6)), pointsPerCloud, noise, jitter, ratioMin, ratioMax, fill);
    const x = U.cloudsToTensor(evalSet.data);
    const y = U.labelsToOneHot(evalSet.labels, U.CLASSES.length);
    const evalRes = await model.evaluate(x, y, { batchSize: 64 });
    const loss = (await evalRes[0].data())[0];
    const accuracy = (await evalRes[1].data())[0];
    const logits = model.predict(x);
    const probs = await logits.array();
    const predIdx = probs.map(row => row.indexOf(Math.max(...row)));
    const conf = U.confusionMatrix(predIdx, evalSet.labels, U.CLASSES.length);
    U.renderEvaluation(evalDiv, { loss, accuracy, confMat: conf });
    evalCache = { data: evalSet.data, labels: evalSet.labels, probs, preds: predIdx, conf, loss, accuracy };
    evalIndex = 0;
    renderCurrent();
    tf.dispose([x, y, logits]);
  }

  function renderCurrent() {
    const mode = viewModeSel.value;
    if (mode === 'eval' && evalCache) {
      const cloud = evalCache.data[evalIndex];
      const trueIdx = evalCache.labels[evalIndex];
      const predIdx = evalCache.preds[evalIndex];
      const prob = Math.max(...evalCache.probs[evalIndex]);
      viewer.render(cloud, trueIdx === predIdx ? '#22c55e' : '#ef4444');
      sampleInfo.textContent = `Eval ${evalIndex+1}/${evalCache.data.length} · True: ${U.CLASSES[trueIdx]} · Pred: ${U.CLASSES[predIdx]} (${(prob*100).toFixed(1)}%)`;
    } else if (dataset.data.length) {
      const cloud = dataset.data[datasetIndex];
      const label = dataset.labels[datasetIndex];
      const color = CLASS_COLOR[U.CLASSES[label]] || '#e6edf3';
      viewer.render(cloud, color);
      sampleInfo.textContent = `Dataset ${datasetIndex+1}/${dataset.data.length} · Class: ${U.CLASSES[label]}`;
    } else {
      sampleInfo.textContent = '';
    }
  }

  function step(direction) {
    const mode = viewModeSel.value;
    if (mode === 'eval' && evalCache) {
      evalIndex = (evalIndex + direction + evalCache.data.length) % evalCache.data.length;
    } else if (dataset.data.length) {
      datasetIndex = (datasetIndex + direction + dataset.data.length) % dataset.data.length;
    }
    renderCurrent();
  }

  // Slider bindings
  noiseSlider.addEventListener('input', () => { noise = parseFloat(noiseSlider.value); noiseValue.textContent = noise.toFixed(2); });
  jitterSlider.addEventListener('input', () => { jitter = parseFloat(jitterSlider.value); jitterValue.textContent = jitter.toFixed(2); });
  pointsSlider.addEventListener('input', () => { pointsPerCloud = parseInt(pointsSlider.value, 10); pointsValue.textContent = String(pointsPerCloud); });
  samplesSlider.addEventListener('input', () => { samplesPerClass = parseInt(samplesSlider.value, 10); samplesValue.textContent = String(samplesPerClass); });
  ratioMinSlider.addEventListener('input', () => {
    ratioMin = Math.max(1, Math.min(6, parseFloat(ratioMinSlider.value)));
    if (ratioMin > ratioMax) { ratioMax = ratioMin; ratioMaxSlider.value = String(ratioMax); }
    ratioMinValue.textContent = ratioMin.toFixed(1);
    ratioMaxValue.textContent = ratioMax.toFixed(1);
  });
  ratioMaxSlider.addEventListener('input', () => {
    ratioMax = Math.max(1, Math.min(6, parseFloat(ratioMaxSlider.value)));
    if (ratioMax < ratioMin) { ratioMin = ratioMax; ratioMinSlider.value = String(ratioMin); }
    ratioMinValue.textContent = ratioMin.toFixed(1);
    ratioMaxValue.textContent = ratioMax.toFixed(1);
  });

  // Button handlers
  uploadInput.addEventListener('change', (ev) => { handleUpload(ev); });
  generateBtn.addEventListener('click', () => { handleGenerate(); });
  trainBtn.addEventListener('click', () => { handleTrain(); });
  evalBtn.addEventListener('click', () => { handleEvaluate(); });
  exportBtn.addEventListener('click', () => { handleExport(); });
  if (fillToggle) fillToggle.addEventListener('change', () => { fill = !!fillToggle.checked; dataset = {data:[],labels:[]}; renderCurrent(); });
  prevBtn.addEventListener('click', () => step(-1));
  nextBtn.addEventListener('click', () => step(1));
  viewModeSel.addEventListener('change', () => renderCurrent());

  // Initialize viewer
  const viewer = U.createOrbitViewer(canvas);

  function renderLegend() {
    const classes = U.CLASSES;
    legendEl.innerHTML = classes.map(c => `<span class="dot ${cssClassFor(c)}"></span><span>${titleFor(c)}</span>`).join(' ');
  }

  function cssClassFor(c){
    switch(c){
      case 'pyramid': return 'pyr';
      case 'box': return 'box';
      case 'cylinder': return 'cyl';
      case 'ellipsoid': return 'ell';
      case 'paraboloid': return 'par';
      case 'cone': return 'cone';
      default: return '';
    }
  }

  function titleFor(c){ return c.charAt(0).toUpperCase()+c.slice(1); }

  function renderClassSelect() {
    const all = U.ALL_CLASSES || ['pyramid','box','cylinder','ellipsoid','paraboloid','cone'];
    const active = new Set(U.getActiveClasses ? U.getActiveClasses() : U.CLASSES);
    classSelect.innerHTML = all.map(c => {
      const id = `cls-${c}`;
      return `<label for="${id}"><input type="checkbox" id="${id}" value="${c}" ${active.has(c)?'checked':''}/> ${titleFor(c)}</label>`;
    }).join(' ');
    for (const input of classSelect.querySelectorAll('input[type=checkbox]')) {
      input.addEventListener('change', onClassToggleChange);
    }
  }

  function onClassToggleChange(){
    const selected = Array.from(classSelect.querySelectorAll('input[type=checkbox]:checked')).map(i => i.value);
    const updated = U.setActiveClasses(selected);
    // Invalidate model if class count changed
    if (model && model.outputs[0].shape[1] !== updated.length) {
      model = null;
    }
    // Regenerate dataset to reflect new label mapping
    dataset = { data: [], labels: [] };
    evalCache = null;
    datasetIndex = 0;
    renderLegend();
    renderCurrent();
  }

  renderClassSelect();
  renderLegend();
})();


