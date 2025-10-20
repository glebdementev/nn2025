// CSV utilities: parsing and export

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  function parseCSVTextToPointCloud(text) {
    const lines = text.trim().split(/\r?\n/);
    const points = [];
    for (const line of lines) {
      if (!line) continue;
      const parts = line.split(',').map(s => Number(s.trim()));
      if (parts.length >= 3 && parts.every(Number.isFinite)) {
        points.push([parts[0], parts[1], parts[2]]);
      }
    }
    return points;
  }

  function processCSVFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const text = String(ev.target.result);
          const cloud = parseCSVTextToPointCloud(text);
          resolve(cloud);
        } catch (e) { reject(e); }
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  function datasetToCSVRows(dataset, classes) {
    // dataset: { data: [ [ [x,y,z], ...P ], ...N ], labels: [classIdx,...] }
    const rows = ['sample_id,class,point_id,x,y,z'];
    for (let i = 0; i < dataset.data.length; i++) {
      const label = classes[dataset.labels[i]];
      const cloud = dataset.data[i];
      for (let j = 0; j < cloud.length; j++) {
        const [x,y,z] = cloud[j];
        rows.push(`${i},${label},${j},${x},${y},${z}`);
      }
    }
    return rows;
  }

  function downloadCSVFromRows(rows, filename) {
    const blob = new Blob([rows.join('\n') + '\n'], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportDatasetCSV(dataset, classes) {
    const rows = datasetToCSVRows(dataset, classes);
    downloadCSVFromRows(rows, 'pointcloud_dataset.csv');
  }

  NS.parseCSVTextToPointCloud = parseCSVTextToPointCloud;
  NS.processCSVFile = processCSVFile;
  NS.exportDatasetCSV = exportDatasetCSV;
})();


