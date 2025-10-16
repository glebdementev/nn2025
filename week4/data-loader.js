class DataLoader {
    constructor() {
        this.stocksData = null;
        this.normalizedData = null;
        this.symbols = [];
        this.dates = [];
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.testDates = [];
        this.featuresPerSymbol = 2;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    this.parseCSV(csv);
                    resolve(this.stocksData);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        
        const data = {};
        const symbols = new Set();
        const dates = new Set();

        // Parse all rows
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length !== headers.length) continue;

            const row = {};
            headers.forEach((header, index) => {
                row[header.trim()] = values[index].trim();
            });

            const symbol = row.Symbol;
            const date = row.Date;
            
            symbols.add(symbol);
            dates.add(date);

            if (!data[symbol]) data[symbol] = {};
            data[symbol][date] = {
                Open: parseFloat(row.Open),
                Close: parseFloat(row.Close),
                High: parseFloat(row.High),
                Low: parseFloat(row.Low),
                Volume: parseFloat(row.Volume)
            };
        }

        this.symbols = Array.from(symbols).sort();
        this.dates = Array.from(dates).sort();
        this.stocksData = data;

        console.log(`Loaded ${this.symbols.length} stocks with ${this.dates.length} trading days`);
    }

    normalizeData() {
        if (!this.stocksData) throw new Error('No data loaded');
        
        this.normalizedData = {};
        const minMax = {};

        // Calculate min-max per stock for Open and Close
        this.symbols.forEach(symbol => {
            minMax[symbol] = {
                Open: { min: Infinity, max: -Infinity },
                Close: { min: Infinity, max: -Infinity }
            };

            this.dates.forEach(date => {
                if (this.stocksData[symbol][date]) {
                    const point = this.stocksData[symbol][date];
                    minMax[symbol].Open.min = Math.min(minMax[symbol].Open.min, point.Open);
                    minMax[symbol].Open.max = Math.max(minMax[symbol].Open.max, point.Open);
                    minMax[symbol].Close.min = Math.min(minMax[symbol].Close.min, point.Close);
                    minMax[symbol].Close.max = Math.max(minMax[symbol].Close.max, point.Close);
                }
            });
        });

        // Normalize data and compute lightweight derived features
        this.symbols.forEach(symbol => {
            this.normalizedData[symbol] = {};
            this.dates.forEach(date => {
                if (this.stocksData[symbol][date]) {
                    const point = this.stocksData[symbol][date];
                    const openNorm = (point.Open - minMax[symbol].Open.min) /
                                     (minMax[symbol].Open.max - minMax[symbol].Open.min);
                    const closeNorm = (point.Close - minMax[symbol].Close.min) /
                                      (minMax[symbol].Close.max - minMax[symbol].Close.min);
                    const denom = point.Open !== 0 ? Math.abs(point.Open) : 1;
                    const retOC = (point.Close - point.Open) / denom;
                    const range = (point.High - point.Low) / denom;

                    this.normalizedData[symbol][date] = {
                        Open: openNorm,
                        Close: closeNorm,
                        RetOC: retOC,
                        Range: range
                    };
                }
            });
        });

        // After initial pass, compute extra rolling-light features per symbol
        this.symbols.forEach(symbol => {
            for (let i = 0; i < this.dates.length; i++) {
                const d0 = this.dates[i];
                const prev1 = i - 1 >= 0 ? this.dates[i - 1] : null;
                const prev2 = i - 2 >= 0 ? this.dates[i - 2] : null;
                if (!this.normalizedData[symbol][d0]) continue;

                const cur = this.stocksData[symbol][d0];
                const hasPrev1 = prev1 && this.stocksData[symbol][prev1];
                const hasPrev2 = prev2 && this.stocksData[symbol][prev2];

                // Ret1: (Close_t - Close_{t-1}) / Close_{t-1}
                const ret1 = hasPrev1 ?
                    (cur.Close - this.stocksData[symbol][prev1].Close) /
                    (this.stocksData[symbol][prev1].Close !== 0 ? Math.abs(this.stocksData[symbol][prev1].Close) : 1)
                    : 0;

                // Mom3: Close_t - Close_{t-3}
                const mom3 = (i - 3 >= 0 && this.stocksData[symbol][this.dates[i - 3]]) ?
                    cur.Close - this.stocksData[symbol][this.dates[i - 3]].Close : 0;

                // VolRatio3: Volume_t / mean(Volume_{t-1..t-3})
                let volMean3 = 0;
                let volCount = 0;
                for (let k = 1; k <= 3; k++) {
                    const idx = i - k;
                    if (idx >= 0) {
                        const dk = this.dates[idx];
                        const pk = this.stocksData[symbol][dk];
                        if (pk) { volMean3 += pk.Volume; volCount++; }
                    }
                }
                volMean3 = volCount > 0 ? volMean3 / volCount : 1;
                const volRatio3 = volMean3 !== 0 ? cur.Volume / volMean3 : 0;

                const bucket = this.normalizedData[symbol][d0];
                bucket.Ret1 = ret1;
                bucket.Mom3 = mom3;
                bucket.VolRatio3 = volRatio3;
            }
        });

        this.featuresPerSymbol = 7; // Open, Close, RetOC, Range, Ret1, Mom3, VolRatio3

        return this.normalizedData;
    }

    createSequences(sequenceLength = 12, predictionHorizon = 3) {
        if (!this.normalizedData) this.normalizeData();

        const sequences = [];
        const targets = [];
        const validDates = [];

        // Create aligned data matrix
        const alignedData = [];
        for (let i = sequenceLength; i < this.dates.length - predictionHorizon; i++) {
            const currentDate = this.dates[i];
            const sequenceData = [];
            let validSequence = true;

            // Get sequence for all symbols
            for (let j = sequenceLength - 1; j >= 0; j--) {
                const seqDate = this.dates[i - j];
                const timeStepData = [];

                this.symbols.forEach(symbol => {
                    if (this.normalizedData[symbol][seqDate]) {
                        const feat = this.normalizedData[symbol][seqDate];
                        timeStepData.push(
                            feat.Open,
                            feat.Close,
                            feat.RetOC,
                            feat.Range
                        );
                    } else {
                        validSequence = false;
                    }
                });

                if (validSequence) sequenceData.push(timeStepData);
            }

            // Create target labels
            if (validSequence) {
                const target = [];
                const baseClosePrices = [];

                // Get base close prices (current date)
                this.symbols.forEach(symbol => {
                    baseClosePrices.push(this.stocksData[symbol][currentDate].Close);
                });

                // Calculate binary labels for prediction horizon
                for (let offset = 1; offset <= predictionHorizon; offset++) {
                    const futureDate = this.dates[i + offset];
                    this.symbols.forEach((symbol, idx) => {
                        if (this.stocksData[symbol][futureDate]) {
                            const futureClose = this.stocksData[symbol][futureDate].Close;
                            target.push(futureClose > baseClosePrices[idx] ? 1 : 0);
                        } else {
                            validSequence = false;
                        }
                    });
                }

                if (validSequence) {
                    sequences.push(sequenceData);
                    targets.push(target);
                    validDates.push(currentDate);
                }
            }
        }

        // Split into train/test (80/20 chronological split)
        const splitIndex = Math.floor(sequences.length * 0.8);
        
        this.X_train = tf.tensor3d(sequences.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(sequences.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
        this.testDates = validDates.slice(splitIndex);

        console.log(`Created ${sequences.length} sequences`);
        console.log(`Training: ${this.X_train.shape[0]}, Test: ${this.X_test.shape[0]}`);
        
        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            symbols: this.symbols,
            testDates: this.testDates,
            featuresPerSymbol: this.featuresPerSymbol
        };
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}
