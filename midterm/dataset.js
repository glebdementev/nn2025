// Dataset utilities: dataset creation, tensors

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  const CLASSES = ["pyramid", "box", "cylinder"];
  NS.CLASSES = CLASSES;

  // Distinct height ranges per class (random within range each sample)
  const HEIGHT_RANGES = {
    pyramid: [0.8, 1.6],
    box: [0.6, 1.2],
    cylinder: [1.0, 2.0],
  };

  function randomHeightFor(shape) {
    const [a,b] = HEIGHT_RANGES[shape];
    return a + Math.random()*(b-a);
  }

  function createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter) {
    const data = [];
    const labels = [];
    for (const [classIndex, shape] of CLASSES.entries()) {
      for (let i=0;i<samplesPerClass;i++) {
        const h = randomHeightFor(shape);
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


