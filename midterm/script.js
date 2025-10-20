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
  const cropShareMinSlider = document.getElementById('crop-share-min-slider');
  const cropShareMaxSlider = document.getElementById('crop-share-max-slider');
  const noiseValue = document.getElementById('noise-value');
  const jitterValue = document.getElementById('jitter-value');
  const pointsValue = document.getElementById('points-value');
  const samplesValue = document.getElementById('samples-value');
  const ratioMinValue = document.getElementById('ratio-min-value');
  const ratioMaxValue = document.getElementById('ratio-max-value');
  const cropShareMinValue = document.getElementById('crop-share-min-value');
  const cropShareMaxValue = document.getElementById('crop-share-max-value');
  const canvas = document.getElementById('point-cloud-canvas');
  const prevBtn = document.getElementById('prev-sample');
  const nextBtn = document.getElementById('next-sample');
  const viewModeSel = document.getElementById('view-mode');
  const sampleInfo = document.getElementById('sample-info');
  const legendEl = document.getElementById('legend');
  const evalDiv = document.getElementById('evaluation-results');
  const logDiv = document.getElementById('training-log');
  const trainChart = document.getElementById('training-chart');
  const trainSummary = document.getElementById('training-summary');

  let noise = parseFloat(noiseSlider.value);
  let jitter = parseFloat(jitterSlider.value);
  let pointsPerCloud = parseInt(pointsSlider.value, 10);
  let samplesPerClass = parseInt(samplesSlider.value, 10);
  let ratioMin = Math.max(1, Math.min(6, parseFloat(ratioMinSlider.value)));
  let ratioMax = Math.max(1, Math.min(6, parseFloat(ratioMaxSlider.value)));
  if (ratioMin > ratioMax) { const tmp = ratioMin; ratioMin = ratioMax; ratioMax = tmp; }
  let cropShareMin = Math.max(0, Math.min(0.9, parseFloat(cropShareMinSlider.value || '0')));
  let cropShareMax = Math.max(0, Math.min(0.9, parseFloat(cropShareMaxSlider.value || '0')));
  if (cropShareMin > cropShareMax) { const t = cropShareMin; cropShareMin = cropShareMax; cropShareMax = t; }
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
    // Keep minimal messages in case needed for debugging
    if (!logDiv) return;
    if (!logDiv._text) logDiv._text = '';
    logDiv._text += (text + '\n');
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
    dataset = U.createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter, ratioMin, ratioMax, fill, cropShareMin, cropShareMax);
    datasetIndex = 0;
    renderCurrent();
  }

  function handleExport() {
    if (!dataset.data.length) return;
    U.exportDatasetCSV(dataset, U.CLASSES);
  }

  async function prepareTensors() {
    if (!dataset.data.length) {
      dataset = U.createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter, ratioMin, ratioMax, fill, cropShareMin, cropShareMax);
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
    const history = { epoch: [], loss: [], acc: [], val_loss: [], val_acc: [] };
    if (trainChart) U.renderTrainingChart(trainChart, history);
    if (trainSummary) U.renderTrainingSummary(trainSummary, history);
    await model.fit(inputs, labels, {
      epochs: 30,
      batchSize: 32,
      validationData: [valInputs, valLbls],
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          history.epoch.push(epoch+1);
          history.loss.push(logs.loss);
          const acc = (logs.acc!=null?logs.acc:logs.accuracy);
          if (acc!=null) (history.acc||(history.accuracy=[])).push(acc);
          history.val_loss.push(logs.val_loss);
          const vacc = (logs.val_acc!=null?logs.val_acc:logs.val_accuracy);
          if (vacc!=null) (history.val_acc||(history.val_accuracy=[])).push(vacc);
          if (trainChart) U.renderTrainingChart(trainChart, history);
          if (trainSummary) U.renderTrainingSummary(trainSummary, history);
        }
      }
    });
    if (trainSummary) U.renderTrainingSummary(trainSummary, history);
  }

  function argMaxPerRow(t) {
    return Array.from(t.argMax(-1).dataSync());
  }

  async function handleEvaluate() {
    if (!model) return;
    // Create a fresh eval set to avoid contamination
    const evalSet = U.createToyDataset(Math.max(60, Math.floor(samplesPerClass*0.6)), pointsPerCloud, noise, jitter, ratioMin, ratioMax, fill, cropShareMin, cropShareMax);
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

  cropShareMinSlider.addEventListener('input', () => {
    cropShareMin = Math.max(0, Math.min(0.9, parseFloat(cropShareMinSlider.value)));
    if (cropShareMin > cropShareMax) { cropShareMax = cropShareMin; cropShareMaxSlider.value = String(cropShareMax); }
    cropShareMinValue.textContent = cropShareMin.toFixed(2);
    cropShareMaxValue.textContent = cropShareMax.toFixed(2);
  });
  cropShareMaxSlider.addEventListener('input', () => {
    cropShareMax = Math.max(0, Math.min(0.9, parseFloat(cropShareMaxSlider.value)));
    if (cropShareMax < cropShareMin) { cropShareMin = cropShareMax; cropShareMinSlider.value = String(cropShareMin); }
    cropShareMinValue.textContent = cropShareMin.toFixed(2);
    cropShareMaxValue.textContent = cropShareMax.toFixed(2);
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


