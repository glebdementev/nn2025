
Role:  
You are a senior front‑end engineer and ML instructor building a browser‑only TensorFlow.js MNIST demo for students.

Context:

-   Build a GitHub Pages–deployable web app that TRAINS and RUNS entirely client‑side with TensorFlow.js and tfjs‑vis.
    
-   MNIST data will be provided by the user as two local files via file inputs: mnist_train.csv and mnist_test.csv.
    
-   CSV format: each row = label (0–9) followed by 784 pixel values (0–255) with no header.
    
-   Do NOT fetch data over the network; parse the two uploaded files in the browser, normalize pixels to [0,1], reshape to [N,28,28,1], and one‑hot labels to depth 10. [tensorflow](https://www.tensorflow.org/tutorials/load_data/csv?hl=ko)
    
-   Implement FILE‑BASED model Save/Load only: download model.json + weights.bin, and reload from user‑selected files (no IndexedDB/LocalStorage).
    
-   Denoising task: add random noise to the test (and training/validation) images and train a CNN autoencoder to reconstruct clean digits.
    
-   Include training with live charts, evaluation of denoising quality, and a preview that displays 5 random test images showing noisy input vs. denoised output side‑by‑side.
    

Instruction:  
Output exactly three fenced code blocks, in this order, labeled “index.html”, “data-loader.js”, and “app.js”, implementing all features below without any extra prose.

-   index.html
    
    -   Include CDNs:
        
        -   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
        -   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
    -   Minimal CSS for a two‑column layout and a horizontal preview strip.
        
    -   Controls:
        
        -   “Upload Train CSV” (<input type="file" id="train-csv" accept=".csv">)
            
        -   “Upload Test CSV” (<input type="file" id="test-csv" accept=".csv">)
            
        -   Buttons: Load Data, Train, Evaluate, Test 5 Random, Save Model (Download), Load Model (From Files), Reset, Toggle Visor.
            
        -   Model load inputs: <input type="file" id="upload-json" accept=".json"> and <input type="file" id="upload-weights" accept=".bin">
            
    -   Sections: Data Status, Training Logs, Metrics (denoising loss/MSE + charts), Random 5 Preview (row where each item shows Noisy vs. Denoised canvases), Model Info (layers/params).
        
    -   Defer‑load data-loader.js then app.js.
        
-   data-loader.js
    
    -   Implement file‑based CSV parsing with FileReader/TextDecoder (no external libraries). Robustly handle large files by chunking or streaming if needed; otherwise readAsText is acceptable.
        
    -   Parse rows as: first value → label int, remaining 784 → pixels; ignore empty lines.
        
    -   Normalize pixels /255, reshape to [N,28,28,1], one‑hot labels depth 10.
        
    -   Add random noise utilities:
        
        -   function addGaussianNoise(xs, std=0.5): returns xsNoisy clipped to [0,1].
        
        -   function makeNoisyCopy(xs, std=0.5): convenience wrapper returning a new tensor.
        
        -   loadTestFromFiles(file): also prepares and stores noisy test tensors {xs, ys, noisyXs}.
        
        -   function getRandomTestDenoiseBatch(cleanXs, noisyXs, k=5): returns matching batches for preview.
        
    -   Provide:
        
        -   async function loadTrainFromFiles(file): returns {xs, ys}
            
        -   async function loadTestFromFiles(file): returns {xs, ys, noisyXs}
            
        -   function splitTrainVal(xs, ys, valRatio=0.1): returns {trainXs, trainYs, valXs, valYs}
            
        -   function getRandomTestBatch(xs, ys, k=5): returns tensors for preview
            
        -   function getRandomTestDenoiseBatch(cleanXs, noisyXs, k=5)
            
        -   function draw28x28ToCanvas(tensor, canvas, scale=4)
            
    -   Dispose intermediate tensors to avoid leaks.
        
-   app.js
    
    -   Wire UI:
        
        -   onLoadData: read both CSV files, build tensors, and show counts. Also create and store noisy test tensors for denoising.
            
        -   onTrain: build and train a CNN autoencoder with tfjs‑vis fitCallbacks using noisy inputs → clean outputs.
            
        -   onEvaluate: compute denoising quality (e.g., MSE) on the noisy test set vs. clean targets; display charts for loss history.
            
        -   onTestFive: sample 5 random test images and display Noisy vs. Denoised canvases for each item in one horizontal row.
            
        -   onSaveDownload: await model.save('downloads://mnist-denoiser')
            
        -   onLoadFromFiles: const m = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile])); replace current model, model.summary(), rebind buttons.
            
        -   onReset: dispose tensors/model and clear UI.
            
    -   Model:
        
        -   CNN Autoencoder (encoder + decoder):  
            Encoder: Conv2D(32,3,'relu','same') → MaxPool2D(2,'same') → Conv2D(64,3,'relu','same') → MaxPool2D(2,'same')  
            Decoder: Conv2DTranspose(64,3,strides=2,'same','relu') → Conv2DTranspose(32,3,strides=2,'same','relu') → Conv2D(1,3,'same','sigmoid')
            
        -   Compile: optimizer='adam', loss='meanSquaredError'.
            
        -   Training defaults: epochs 5–10, batchSize 64–128, shuffle true; record duration and best val loss.
            
    -   Charts (tfjs‑vis):
        
        -   Live loss/val_loss during fit.
            
        -   Show scalar metrics for denoising evaluation (e.g., MSE).
            
    -   Performance & safety:
        
        -   Use tf.tidy where appropriate; dispose old models/tensors on replace.
            
        -   Try/catch around file handling and training; show friendly error messages.
            
        -   Ensure UI stays responsive (await/queueMicrotask, requestAnimationFrame for long operations).
            

Formatting:

-   Produce only three fenced code blocks labeled exactly “index.html”, “data-loader.js”, and “app.js”.
    
-   Browser‑only JavaScript; no Node or extra libraries; clear English comments; no text outside the code blocks.

-   Provide very detail and intuitive comments as explanation for important code blocks.
