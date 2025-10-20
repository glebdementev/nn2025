// Dataset utilities: dataset creation, tensors

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  const CLASSES = ["pyramid", "box", "cylinder"];
  NS.CLASSES = CLASSES;

  // Height determined by height-to-width ratio r in [ratioMin, ratioMax]
  // Base width across x,y is 2 (from -1 to 1), so height = r * 2
  function randomHeightForRatio(ratioMin, ratioMax) {
    const rMin = Math.max(1, Math.min(6, ratioMin));
    const rMax = Math.max(1, Math.min(6, ratioMax));
    const lo = Math.min(rMin, rMax);
    const hi = Math.max(rMin, rMax);
    const r = lo + Math.random()*(hi - lo);
    return r * 2;
  }

  function createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter, ratioMin=1, ratioMax=3) {
    const data = [];
    const labels = [];
    for (const [classIndex, shape] of CLASSES.entries()) {
      for (let i=0;i<samplesPerClass;i++) {
        const h = randomHeightForRatio(ratioMin, ratioMax);
        const cloud = window.PointCloudUtils.generateShape(shape, pointsPerCloud, noise, jitter, h);
        data.push(cloud);
        labels.push(classIndex);
      }
    }
    return { data, labels };
  }

  function cloudsToTensor(clouds) {
    const flat = [];
    for (const cloud of clouds) {
      for (const p of cloud) flat.push(p[0], p[1], p[2]);
    }
    const n = clouds.length;
    const p = clouds[0]?.length || 0;
    return tf.tensor(flat, [n, p, 3]);
  }

  function labelsToOneHot(labels, numClasses) {
    const n = labels.length;
    const arr = new Float32Array(n * numClasses);
    for (let i=0;i<n;i++) arr[i*numClasses + labels[i]] = 1;
    return tf.tensor(arr, [n, numClasses]);
  }

  function shuffleInUnison(a, b) {
    for (let i=a.length-1;i>0;i--) {
      const j = Math.floor(Math.random()*(i+1));
      [a[i], a[j]] = [a[j], a[i]];
      [b[i], b[j]] = [b[j], b[i]];
    }
  }

  function splitTrainVal(data, labels, valRatio=0.2) {
    const idx = data.map((_,i)=>i);
    shuffleInUnison(idx, idx);
    const nVal = Math.max(1, Math.floor(data.length * valRatio));
    const valSet = new Set(idx.slice(0, nVal));
    const trainData=[], trainLabels=[], valData=[], valLabels=[];
    for (let i=0;i<data.length;i++) {
      if (valSet.has(i)) { valData.push(data[i]); valLabels.push(labels[i]); }
      else { trainData.push(data[i]); trainLabels.push(labels[i]); }
    }
    return { trainData, trainLabels, valData, valLabels };
  }

  NS.createToyDataset = createToyDataset;
  NS.cloudsToTensor = cloudsToTensor;
  NS.labelsToOneHot = labelsToOneHot;
  NS.splitTrainVal = splitTrainVal;
})();


