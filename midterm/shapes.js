// Shape generators: pyramid, box, cylinder with variable heights

(function(){
  const NS = window.PointCloudUtils || (window.PointCloudUtils = {});

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function addNoiseJitter(point, noise, jitter) {
    const [x,y,z] = point;
    const jx = (Math.random()*2-1)*jitter;
    const jy = (Math.random()*2-1)*jitter;
    const jz = (Math.random()*2-1)*jitter;
    return [x + randn()*noise + jx, y + randn()*noise + jy, z + randn()*noise + jz];
  }

  function centerOnly(points) {
    if (!points.length) return points;
    let cx=0,cy=0,cz=0;
    for (const [x,y,z] of points) { cx+=x; cy+=y; cz+=z; }
    cx/=points.length; cy/=points.length; cz/=points.length;
    return points.map(([x,y,z]) => [x-cx, y-cy, z-cz]);
  }

  function generatePyramid(pointsCount, height) {
    // Square base [-1,1]^2 at z=0, apex at z=height
    const pts = [];
    const apex = [0,0,height];
    const faces = [
      { a: apex, b: [-1,-1,0], c: [1,-1,0] },
      { a: apex, b: [1,-1,0], c: [1,1,0] },
      { a: apex, b: [1,1,0], c: [-1,1,0] },
      { a: apex, b: [-1,1,0], c: [-1,-1,0] },
    ];
    const baseEdges = [
      [[-1,-1,0],[1,-1,0]], [[1,-1,0],[1,1,0]], [[1,1,0],[-1,1,0]], [[-1,1,0],[-1,-1,0]]
    ];
    for (let i=0;i<pointsCount;i++) {
      const r = Math.random();
      if (r < 0.7) {
        const f = faces[Math.floor(Math.random()*faces.length)];
        let u = Math.random(); let v = Math.random();
        if (u+v>1) { u=1-u; v=1-v; }
        const w = 1-u-v;
        const x = f.a[0]*u + f.b[0]*v + f.c[0]*w;
        const y = f.a[1]*u + f.b[1]*v + f.c[1]*w;
        const z = f.a[2]*u + f.b[2]*v + f.c[2]*w;
        pts.push([x,y,z]);
      } else {
        const [p0,p1] = baseEdges[Math.floor(Math.random()*baseEdges.length)];
        const t = Math.random();
        pts.push([
          p0[0] + t*(p1[0]-p0[0]),
          p0[1] + t*(p1[1]-p0[1]),
          0
        ]);
      }
    }
    return pts;
  }

  function generateBox(pointsCount, height) {
    // AABB with x,y in [-1,1], z in [-height/2, height/2]
    const pts = [];
    const zmin = -height/2, zmax = height/2;
    const faces = [
      {z:zmax}, {z:zmin}, {x:1}, {x:-1}, {y:1}, {y:-1}
    ];
    for (let i=0;i<pointsCount;i++) {
      const f = faces[Math.floor(Math.random()*faces.length)];
      let x=0,y=0,z=0;
      if ('z' in f) { z=f.z; x=Math.random()*2-1; y=Math.random()*2-1; }
      else if ('x' in f) { x=f.x; y=Math.random()*2-1; z=Math.random()*(zmax-zmin)+zmin; }
      else { y=f.y; x=Math.random()*2-1; z=Math.random()*(zmax-zmin)+zmin; }
      if (Math.random()<0.4) {
        if (Math.random()<0.5) x = Math.round(x);
        else if (Math.random()<0.5) y = Math.round(y);
        else z = Math.abs(z - zmin) < Math.abs(z - zmax) ? zmin : zmax;
      }
      pts.push([x,y,z]);
    }
    return pts;
  }

  function generateCylinder(pointsCount, height) {
    // Cylinder radius 1, z in [-height/2, height/2]
    const pts = [];
    const zmin = -height/2, zmax = height/2;
    for (let i=0;i<pointsCount;i++) {
      const r = Math.random();
      if (r < 0.75) {
        const theta = Math.random()*Math.PI*2;
        const z = Math.random()*(zmax - zmin) + zmin;
        pts.push([Math.cos(theta), Math.sin(theta), z]);
      } else {
        const theta = Math.random()*Math.PI*2;
        const z = Math.random()<0.5 ? zmin : zmax;
        pts.push([Math.cos(theta), Math.sin(theta), z]);
      }
    }
    return pts;
  }

  function generateShape(shape, pointsCount, noise, jitter, height) {
    let pts;
    switch (shape) {
      case 'pyramid': pts = generatePyramid(pointsCount, height); break;
      case 'box': pts = generateBox(pointsCount, height); break;
      case 'cylinder': pts = generateCylinder(pointsCount, height); break;
      default: pts = generateBox(pointsCount, height);
    }
    const noisy = pts.map(p => addNoiseJitter(p, noise, jitter));
    return centerOnly(noisy); // center only, no scaling (preserve height differences)
  }

  NS.generateShape = generateShape;
})();


