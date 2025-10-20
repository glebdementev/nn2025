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

  function generatePyramid(pointsCount, height, fill=false) {
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
      if (fill) {
        // Uniform in volume: z = h*(1 - u^{1/3}), x,y in square scaled by (1 - z/h)
        const u = Math.random();
        const z = height * (1 - Math.cbrt(u));
        const scale = (1 - z/height);
        const x = (Math.random()*2 - 1) * scale;
        const y = (Math.random()*2 - 1) * scale;
        pts.push([x,y,z]);
      } else {
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
    }
    return pts;
  }

  function generateBox(pointsCount, height, fill=false) {
    // AABB with x,y in [-1,1], z in [-height/2, height/2]
    const pts = [];
    const zmin = -height/2, zmax = height/2;
    const faces = [
      {z:zmax}, {z:zmin}, {x:1}, {x:-1}, {y:1}, {y:-1}
    ];
    for (let i=0;i<pointsCount;i++) {
      if (fill) {
        const x = Math.random()*2 - 1;
        const y = Math.random()*2 - 1;
        const z = Math.random()*(zmax - zmin) + zmin;
        pts.push([x,y,z]);
      } else {
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
    }
    return pts;
  }

  function generateCylinder(pointsCount, height, fill=false) {
    // Cylinder radius 1, z in [-height/2, height/2]
    const pts = [];
    const zmin = -height/2, zmax = height/2;
    for (let i=0;i<pointsCount;i++) {
      if (fill) {
        const z = Math.random()*(zmax - zmin) + zmin;
        const r = Math.sqrt(Math.random());
        const theta = Math.random()*Math.PI*2;
        pts.push([r*Math.cos(theta), r*Math.sin(theta), z]);
      } else {
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
    }
    return pts;
  }

  function generateEllipsoid(pointsCount, height, fill=false) {
    // Ellipsoid with semi-axes a=b=1 in x,y and c=height/2 in z
    const pts = [];
    const a = 1, b = 1, c = height/2;
    for (let i=0;i<pointsCount;i++) {
      if (fill) {
        // Sample uniformly inside ellipsoid: radius^(1/3) times random unit vector
        // Random direction via normal distribution
        let xg = randn(), yg = randn(), zg = randn();
        const norm = Math.hypot(xg, yg, zg) || 1;
        xg /= norm; yg /= norm; zg /= norm;
        const r = Math.cbrt(Math.random());
        const x = a * r * xg;
        const y = b * r * yg;
        const z = c * r * zg;
        pts.push([x,y,z]);
      } else {
        // Approx uniform on surface using sphere parameterization
        const u = Math.random()*2 - 1; // cos(phi)
        const theta = Math.random()*Math.PI*2;
        const s = Math.sqrt(1 - u*u);
        const x = a * s * Math.cos(theta);
        const y = b * s * Math.sin(theta);
        const z = c * u;
        pts.push([x,y,z]);
      }
    }
    return pts;
  }

  function generateParaboloid(pointsCount, height, fill=false) {
    // Downward-opening paraboloid: z = -height * r^2 with r in [0,1], truncated at z in [-height, 0]
    const pts = [];
    for (let i=0;i<pointsCount;i++) {
      if (fill) {
        // Uniform in volume under paraboloid cap: sample z with CDF ~ (|z|/h)^2 => z = -h*sqrt(u)
        const u = Math.random();
        const z = -height * Math.sqrt(u);
        const rMax = Math.sqrt(Math.abs(z)/height); // = u^{1/4}
        const r = rMax * Math.sqrt(Math.random());
        const theta = Math.random()*Math.PI*2;
        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta);
        pts.push([x,y,z]);
      } else {
        const r = Math.sqrt(Math.random()); // more points near rim
        const theta = Math.random()*Math.PI*2;
        const z = -height * (r*r);
        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta);
        pts.push([x,y,z]);
      }
    }
    return pts;
  }

  function generateCone(pointsCount, height, fill=false) {
    // Right circular cone with base radius 1 at z=0 and apex at z=height
    const pts = [];
    for (let i=0;i<pointsCount;i++) {
      if (fill) {
        // Uniform inside cone: z = h*(1 - u^{1/3}), radius = (1 - z/h) * sqrt(v)
        const u = Math.random();
        const z = height * (1 - Math.cbrt(u));
        const baseR = 1 - (z/height);
        const r = baseR * Math.sqrt(Math.random());
        const theta = Math.random()*Math.PI*2;
        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta);
        pts.push([x,y,z]);
      } else {
        const r = Math.random();
        if (r < 0.8) {
          // Lateral surface
          const z = Math.random()*height;
          const radius = 1 - (z/height);
          const theta = Math.random()*Math.PI*2;
          const x = radius * Math.cos(theta);
          const y = radius * Math.sin(theta);
          pts.push([x,y,z]);
        } else {
          // Base circle at z=0
          const theta = Math.random()*Math.PI*2;
          pts.push([Math.cos(theta), Math.sin(theta), 0]);
        }
      }
    }
    return pts;
  }

  function generateShape(shape, pointsCount, noise, jitter, height, fill=false) {
    let pts;
    switch (shape) {
      case 'pyramid': pts = generatePyramid(pointsCount, height, fill); break;
      case 'box': pts = generateBox(pointsCount, height, fill); break;
      case 'cylinder': pts = generateCylinder(pointsCount, height, fill); break;
      case 'ellipsoid': pts = generateEllipsoid(pointsCount, height, fill); break;
      case 'paraboloid': pts = generateParaboloid(pointsCount, height, fill); break;
      case 'cone': pts = generateCone(pointsCount, height, fill); break;
      default: pts = generateBox(pointsCount, height);
    }
    const noisy = pts.map(p => addNoiseJitter(p, noise, jitter));
    return centerOnly(noisy); // center only, no scaling (preserve height differences)
  }

  NS.generateShape = generateShape;
})();


