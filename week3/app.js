class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        this.initializeUI();
    }

    initializeUI() {
        // Bind button events
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
    }

    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                this.showError('Please select both train and test CSV files');
                return;
            }

            this.showStatus('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.showStatus('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.trainData = trainData;
            this.testData = testData;

            this.updateDataStatus(trainData.count, testData.count);
            this.showStatus('Data loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    async onTrain() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.showStatus('Starting training...');
            
            // Split training data (targets are clean images for denoising)
            const { trainXs, valXs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );
            // Create noisy inputs matching the splits
            const noisyTrainXs = this.dataLoader.makeNoisyCopy(trainXs, 0.3);
            const noisyValXs = this.dataLoader.makeNoisyCopy(valXs, 0.3);

            // Create or get model
            if (!this.model) {
                this.model = this.createDenoiserModel();
                this.updateModelInfo();
            }

            // Train with tfjs-vis callbacks
            const startTime = Date.now();
            const history = await this.model.fit(noisyTrainXs, trainXs, {
                epochs: 3,
                batchSize: 64,
                validationData: [noisyValXs, valXs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Denoiser Training' },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            const duration = (Date.now() - startTime) / 1000;
            const bestValLoss = Math.min(...history.history.val_loss);
            
            this.showStatus(`Training completed in ${duration.toFixed(1)}s. Best val_loss: ${bestValLoss.toFixed(4)}`);
            
            // Clean up
            trainXs.dispose();
            valXs.dispose();
            noisyTrainXs.dispose();
            noisyValXs.dispose();
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    async onEvaluate() {
        if (!this.model) {
            this.showError('No model available. Please train or load a model first.');
            return;
        }
        if (!this.testData || !this.testData.noisyXs) {
            this.showError('No noisy test data available');
            return;
        }
        try {
            this.showStatus('Evaluating denoiser (masked MSE on test set)...');
            const denoised = this.model.predict(this.testData.noisyXs);
            const masked = this.maskedMSE(this.testData.xs, denoised);
            const mse = (await masked.data())[0];
            tfvis.show.modelSummary({ name: 'Denoiser Model', tab: 'Evaluation' }, this.model);
            this.showStatus(`Test masked MSE: ${mse.toFixed(6)}`);
            denoised.dispose();
            masked.dispose();
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    async onTestFive() {
        if (!this.model || !this.testData || !this.testData.noisyXs) {
            this.showError('Please load model and test data first');
            return;
        }
        try {
            const { batchClean, batchNoisy } = this.dataLoader.getRandomTestDenoiseBatch(
                this.testData.xs, this.testData.noisyXs, 5
            );
            const denoised = this.model.predict(batchNoisy);
            await this.renderDenoisePreview(batchNoisy, denoised, batchClean);
            denoised.dispose();
            batchClean.dispose();
            batchNoisy.dispose();
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    async onSaveDownload() {
        if (!this.model) {
            this.showError('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-denoiser');
            this.showStatus('Model saved successfully!');
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...');
            
            // Dispose old model if exists
            if (this.model) {
                this.model.dispose();
            }
            
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            this.showStatus('Model loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    createDenoiserModel() {
        const model = tf.sequential();
        // Encoder
        model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1] }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: 'same' }));
        // Decoder using upsampling + conv for broader TF.js support
        model.add(tf.layers.upSampling2d({ size: [2, 2] }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        model.add(tf.layers.upSampling2d({ size: [2, 2] }));
        model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
        model.add(tf.layers.conv2d({ filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same' }));
        model.compile({ optimizer: tf.train.adam(0.001), loss: (yTrue, yPred) => this.maskedMSE(yTrue, yPred) });
        return model;
    }

    // Weighted binary cross-entropy to upweight positive (ink) pixels
    weightedBCE(yTrue, yPred) {
        const eps = tf.scalar(1e-7);
        const one = tf.scalar(1);
        const yPredClipped = yPred.clipByValue(1e-7, 1 - 1e-7);
        const posWeight = tf.scalar(5.0); // emphasize foreground strokes
        const weight = yTrue.mul(posWeight).add(one.sub(yTrue));
        const bce = yTrue.mul(yPredClipped.log()).add(one.sub(yTrue).mul(one.sub(yPredClipped).log()));
        const loss = bce.mul(weight).mul(tf.scalar(-1)).mean();
        return loss;
    }

    // Foreground-masked MSE: ignore zero-valued background pixels
    maskedMSE(yTrue, yPred) {
        return tf.tidy(() => {
            const zero = tf.scalar(0);
            const eps = tf.scalar(1e-7);
            const mask = yTrue.greater(zero).cast('float32');
            const se = yPred.sub(yTrue).square().mul(mask);
            const denom = mask.sum().add(eps);
            return se.sum().div(denom);
        });
    }

    async calculateAccuracy(predicted, trueLabels) {
        const equals = predicted.equal(trueLabels);
        const accuracy = equals.mean();
        const result = await accuracy.data();
        equals.dispose();
        accuracy.dispose();
        return result[0];
    }

    async createConfusionMatrix(predicted, trueLabels) {
        const predArray = await predicted.array();
        const trueArray = await trueLabels.array();
        
        const matrix = Array(10).fill().map(() => Array(10).fill(0));
        
        for (let i = 0; i < predArray.length; i++) {
            const pred = predArray[i];
            const trueVal = trueArray[i];
            matrix[trueVal][pred]++;
        }
        
        return matrix;
    }

    calculatePerClassAccuracy(confusionMatrix) {
        return confusionMatrix.map((row, i) => {
            const correct = row[i];
            const total = row.reduce((sum, val) => sum + val, 0);
            return total > 0 ? correct / total : 0;
        });
    }

    async renderDenoisePreview(noisyBatch, denoisedBatch, cleanBatch) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        const noisyArr = noisyBatch.arraySync();
        const denoisedArr = denoisedBatch.arraySync();
        const cleanArr = cleanBatch.arraySync();
        for (let i = 0; i < noisyArr.length; i++) {
            const item = document.createElement('div');
            item.className = 'preview-item';
            const row = document.createElement('div');
            row.className = 'preview-row';
            const noisyCanvas = document.createElement('canvas');
            const denoisedCanvas = document.createElement('canvas');
            const cleanCanvas = document.createElement('canvas');
            this.dataLoader.draw28x28ToCanvas(tf.tensor(noisyArr[i]), noisyCanvas, 4);
            this.dataLoader.draw28x28ToCanvas(tf.tensor(denoisedArr[i]), denoisedCanvas, 4);
            this.dataLoader.draw28x28ToCanvas(tf.tensor(cleanArr[i]), cleanCanvas, 4);
            const caption = document.createElement('div');
            caption.textContent = 'Noisy | Denoised | Clean';
            row.appendChild(noisyCanvas);
            row.appendChild(denoisedCanvas);
            row.appendChild(cleanCanvas);
            item.appendChild(row);
            item.appendChild(caption);
            container.appendChild(item);
        }
    }

    clearPreview() {
        document.getElementById('previewContainer').innerHTML = '';
    }

    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        statusEl.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        
        if (!this.model) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
            return;
        }
        
        let totalParams = 0;
        this.model.layers.forEach(layer => {
            layer.getWeights().forEach(weight => {
                totalParams += weight.size;
            });
        });
        
        infoEl.innerHTML = `
            <h3>Model Info</h3>
            <p>Layers: ${this.model.layers.length}</p>
            <p>Total parameters: ${totalParams.toLocaleString()}</p>
        `;
    }

    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
    }

    showError(message) {
        this.showStatus(`ERROR: ${message}`);
        console.error(message);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
