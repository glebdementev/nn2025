// Dataset utilities: dataset creation, tensors

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  const ALL_CLASSES = ["pyramid", "box", "cylinder", "ellipsoid", "paraboloid", "cone"];
  let ACTIVE_CLASSES = ["pyramid", "box", "cylinder"];
  NS.ALL_CLASSES = ALL_CLASSES;
  NS.CLASSES = ACTIVE_CLASSES;

  // Height determined by height-to-width ratio r in [ratioMin, ratioMax]
  // Base width across x,y is 2 (from -1 to 1), so height = r * 2
  function randomHeightForRatio(ratioMin, ratioMax) {
    const rMin = Math.max(3, Math.min(10, ratioMin));
    const rMax = Math.max(3, Math.min(10, ratioMax));
    const lo = Math.min(rMin, rMax);
    const hi = Math.max(rMin, rMax);
    const r = lo + Math.random()*(hi - lo);
    return r * 2;
  }

  function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

  function ensureSamePointCount(cloud, targetPoints) {
    const out = [];
    if (cloud.length === 0) return Array.from({length:targetPoints}, () => [0,0,0]);
    if (cloud.length >= targetPoints) {
      for (let i=0;i<targetPoints;i++) out.push(cloud[Math.floor(Math.random()*cloud.length)]);
    } else {
      for (let i=0;i<targetPoints;i++) out.push(cloud[i % cloud.length]);
    }
    return out;
  }

  // Crop points by an axis-aligned rectangle in XY of a given area share s in [0,0.9]
  // Base XY domain is [-1,1]x[-1,1] with area 4. Rectangle area = s*4.
  function cropCloudXY(cloud, shareMin, shareMax) {
    const sMin = clamp(shareMin ?? 0, 0, 0.9);
    const sMax = clamp(shareMax ?? 0, 0, 0.9);
    if (sMax <= 0 || sMin < 0 || sMin > 0.9) return cloud;
    const lo = Math.min(sMin, sMax);
    const hi = Math.max(sMin, sMax);
    const tries = 5;
    for (let t=0;t<tries;t++) {
      const s = lo + Math.random()*(hi - lo);
      // Choose width W in [2s, 2], height H = (4s)/W so that W*H = 4s and both <= 2
      const wMin = Math.max(0.0001, 2*s);
      const W = wMin + Math.random()*(2 - wMin);
      const H = (4*s) / W;
      const cx = (Math.random()*2 - 1) * (1 - W/2);
      const cy = (Math.random()*2 - 1) * (1 - H/2);
      const x0 = cx - W/2, x1 = cx + W/2;
      const y0 = cy - H/2, y1 = cy + H/2;
      const cropped = cloud.filter(([x,y,_z]) => x>=x0 && x<=x1 && y>=y0 && y<=y1);
      if (cropped.length > 0) return cropped;
    }
    // Fallback if rectangle missed all points repeatedly: return original
    return cloud;
  }

  function createToyDataset(samplesPerClass, pointsPerCloud, noise, jitter, ratioMin=3, ratioMax=10, fill=false, cropShareMin=0, cropShareMax=0) {
    const data = [];
    const labels = [];
    for (let i=0;i<samplesPerClass;i++) {
      for (const [classIndex, shape] of ACTIVE_CLASSES.entries()) {
        const h = randomHeightForRatio(ratioMin, ratioMax);
        let cloud = window.PointCloudUtils.generateShape(shape, pointsPerCloud, noise, jitter, h, fill);
        if ((cropShareMin>0 || cropShareMax>0)) {
          cloud = cropCloudXY(cloud, cropShareMin, cropShareMax);
        }
        cloud = ensureSamePointCount(cloud, pointsPerCloud);
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
  NS.setActiveClasses = function(selected) {
    const unique = Array.from(new Set((selected||[]).filter(s => ALL_CLASSES.includes(s))));
    if (unique.length === 0) return ACTIVE_CLASSES; // ignore empty selections
    ACTIVE_CLASSES = unique.slice();
    NS.CLASSES = ACTIVE_CLASSES;
    return ACTIVE_CLASSES;
  };
  NS.getActiveClasses = function(){ return ACTIVE_CLASSES.slice(); };
})();


