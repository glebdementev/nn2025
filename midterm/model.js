// Tiny PointNet-style model with shared MLP + global max pooling

function buildPointNetTiny(pointsPerCloud, numClasses) {
  // Input: [B, P, 3]
  const input = tf.input({shape: [pointsPerCloud, 3]});

  // Shared MLP via 1x1 convs (Dense applied TimeDistributed is not native; use conv1d)
  let x = tf.layers.conv1d({filters: 64, kernelSize: 1, activation: 'relu'}).apply(input);
  x = tf.layers.conv1d({filters: 128, kernelSize: 1, activation: 'relu'}).apply(x);
  x = tf.layers.conv1d({filters: 256, kernelSize: 1, activation: 'relu'}).apply(x);

  // Global max pool over points
  x = tf.layers.globalMaxPooling1d().apply(x);

  // Fully connected classifier
  x = tf.layers.dense({units: 128, activation: 'relu'}).apply(x);
  x = tf.layers.dropout({rate: 0.2}).apply(x);
  x = tf.layers.dense({units: 64, activation: 'relu'}).apply(x);
  const output = tf.layers.dense({units: numClasses, activation: 'softmax'}).apply(x);

  const model = tf.model({inputs: input, outputs: output});
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

window.PointNetTiny = { buildPointNetTiny };


