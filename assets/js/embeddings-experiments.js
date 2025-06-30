// util function for seeded random numbers
function mulberry32(a) {
    return function() {
        var t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

class DataGenerator {
    constructor() {
        this.datasets = {};
    }

    generateRandomData(nPoints = 100, nDimensions = 50) {
        const data = [];
        for (let i = 0; i < nPoints; i++) {
            const point = {
                id: i,
                label: `Point ${i}`,
                features: []
            };
            
            for (let d = 0; d < nDimensions; d++) {
                point.features.push(this.randomNormal() * 3);
            }
            
            data.push(point);
        }
        
        return {
            points: data,
            metadata: {
                type: 'random',
                dimensions: nDimensions,
                description: 'Pure random noise with no inherent structure'
            }
        };
    }

    generateClusteredData(nPoints = 200, nDimensions = 50, nClusters = 4) {
        const data = [];
        const pointsPerCluster = Math.floor(nPoints / nClusters);

        const clusterCenters = [];
        for (let c = 0; c < nClusters; c++) {
            const center = [];
            for (let d = 0; d < nDimensions; d++) {
                center.push(c * 1.0 + Math.random() * 0.3); // Instead of c * 15 + Math.random() * 2
            }
            clusterCenters.push(center);
        }

        let pointId = 0;
        for (let c = 0; c < nClusters; c++) {
            for (let p = 0; p < pointsPerCluster; p++) {
                const point = {
                    id: pointId++,
                    label: `Cluster ${c} - Point ${p}`,
                    cluster: c,
                    features: []
                };

                for (let d = 0; d < nDimensions; d++) {
                    point.features.push(
                        clusterCenters[c][d] + this.randomNormal() * 2.0 // Instead of 0.3, more inter-cluster variance
                    );
                }
                
                data.push(point);
            }
        }

        while (data.length < nPoints) {
            const c = Math.floor(Math.random() * nClusters);
            const point = {
                id: pointId++,
                label: `Cluster ${c} - Point ${data.length}`,
                cluster: c,
                features: []
            };
            
            for (let d = 0; d < nDimensions; d++) {
                point.features.push(
                    clusterCenters[c][d] + this.randomNormal() * 0.3
                );
            }
            
            data.push(point);
        }
        
        return {
            points: data,
            metadata: {
                type: 'clustered',
                dimensions: nDimensions,
                clusters: nClusters,
                description: 'Data with clear cluster structure'
            }
        };
    }

    randomNormal() {
        // Box-Muller transform for normal distribution
        if (this.spare !== undefined) {
            const val = this.spare;
            delete this.spare;
            return val;
        }
        
        const u = Math.random();
        const v = Math.random();
        const mag = Math.sqrt(-2 * Math.log(u));
        const norm = mag * Math.cos(2 * Math.PI * v);
        this.spare = mag * Math.sin(2 * Math.PI * v);
        
        return norm;
    }

    generateManifoldData(nPoints = 120, nDimensions = 25) {
        const data = [];
        
        for (let i = 0; i < nPoints; i++) {
            const t = (i / nPoints) * 6 * Math.PI;
            const point = {
                id: i,
                label: `Point ${i}`,
                position: t,
                features: []
            };

            const radius = t / 15;
            point.features.push(Math.cos(t) * radius);
            point.features.push(Math.sin(t) * radius);
            point.features.push(t / 20);

            for (let d = 3; d < nDimensions; d++) {
                if (d % 4 === 0) {
                    point.features.push(Math.sin(t / 3) * 0.5 + this.randomNormal() * 0.1);
                } else if (d % 4 === 1) {
                    point.features.push(Math.cos(t / 4) * 0.3 + this.randomNormal() * 0.1);
                } else {
                    point.features.push(this.randomNormal() * 0.1);
                }
            }
            
            data.push(point);
        }
        
        return {
            points: data,
            metadata: {
                type: 'manifold',
                dimensions: nDimensions,
                description: 'Spiral manifold in high dimensions'
            }
        };
    }

    generateAmbiguousData(nPoints = 180, nDimensions = 40, nClusters = 3) {
        const data = [];
        const pointsPerCluster = Math.floor(nPoints / nClusters);

        const clusterCenters = [];
        for (let c = 0; c < nClusters; c++) {
            const center = [];
            for (let d = 0; d < nDimensions; d++) {
                center.push(c * 2.5 + this.randomNormal() * 0.3);
            }
            clusterCenters.push(center);
        }

        let pointId = 0;
        for (let c = 0; c < nClusters; c++) {
            for (let p = 0; p < pointsPerCluster; p++) {
                const point = {
                    id: pointId++,
                    label: `Point ${pointId-1}`,
                    cluster: c,
                    features: []
                };

                for (let d = 0; d < nDimensions; d++) {
                    point.features.push(
                        clusterCenters[c][d] + this.randomNormal() * 1.2 // more noise
                    );
                }
                
                data.push(point);
            }
        }

        while (data.length < nPoints) {
            const c = Math.floor(Math.random() * nClusters);
            const point = {
                id: pointId++,
                label: `Point ${pointId-1}`,
                cluster: c,
                features: []
            };
            
            for (let d = 0; d < nDimensions; d++) {
                point.features.push(
                    clusterCenters[c][d] + this.randomNormal() * 1.2
                );
            }
            
            data.push(point);
        }
        
        return {
            points: data,
            metadata: {
                type: 'ambiguous-clusters',
                dimensions: nDimensions,
                clusters: nClusters,
                description: 'Overlapping clusters - the kind researchers actually encounter'
            }
        };
    }
}

class DataGeneratorOld {
    constructor() {
        this.datasets = {};
    }

    generateRandomData(nPoints = 200, nDimensions = 20) {
        const data = [];
        for (let i = 0; i < nPoints; i++) {
            const point = {
                id: i,
                label: `Point ${i}`,
                features: []
            };
            
            for (let d = 0; d < nDimensions; d++) {
                point.features.push(Math.random());
            }
            
            data.push(point);
        }
        
        return {
            points: data,
            metadata: {
                type: 'random',
                dimensions: nDimensions,
                description: 'random data with no inherent structure'
            }
        };
    }

    // w/ actual clusters
    generateClusteredData(nPoints = 200, nDimensions = 50, nClusters = 4) {
        const data = [];
        const pointsPerCluster = Math.floor(nPoints / nClusters);

        const clusterCenters = [];
        for (let c = 0; c < nClusters; c++) {
            const center = [];
            for (let d = 0; d < nDimensions; d++) {
                // centers roughly 10 units apart + jitter
                center.push(c * 4 + Math.random() * 0.5);
                // EDIT: 3-5 unit sep
                // center.push(c * 3 + Math.random() * 2);
            }
            clusterCenters.push(center);
        }
        
        // points around each center
        let pointId = 0;
        for (let c = 0; c < nClusters; c++) {
            for (let p = 0; p < pointsPerCluster; p++) {
                const point = {
                    id: pointId++,
                    label: `Cluster ${c} - Point ${p}`,
                    cluster: c,
                    features: []
                };
                
                // noise around cluster center
                for (let d = 0; d < nDimensions; d++) {
                    point.features.push(
                        clusterCenters[c][d] + (Math.random() - 0.5) * 0.5
                    );
                }
                
                data.push(point);
            }
        }
        
        return {
            points: data,
            metadata: {
                type: 'clustered',
                dimensions: nDimensions,
                clusters: nClusters,
                description: 'Data with true cluster structure'
            }
        };
    }

    // continuous manifold data e.g. spiral
    generateManifoldData(nPoints = 200, nDimensions = 20) {
        const data = [];
        
        for (let i = 0; i < nPoints; i++) {
            const t = (i / nPoints) * 4 * Math.PI;
            const point = {
                id: i,
                label: `Point ${i}`,
                position: t, // position along manifold
                features: []
            };
            
            // pattern for first few dims
            point.features.push(Math.cos(t) * t / 10);
            point.features.push(Math.sin(t) * t / 10);
            point.features.push(t / 10);
            
            // small random noise for rest
            for (let d = 3; d < nDimensions; d++) {
                point.features.push(Math.random() * 0.1);
            }
            
            data.push(point);
        }
        
        return {
            points: data,
            metadata: {
                type: 'manifold',
                dimensions: nDimensions,
                description: 'Continuous manifold (spiral) in high dimensions'
            }
        };
    }
}

class EmbeddingAlgorithms {
    static tSNE(data, perplexity = 30, iterations = 500, seed = null) {
        const random = seed != null ? mulberry32(seed) : Math.random;
        const nPoints = data.points.length;
        const features = data.points.map(p => p.features);

        console.log(`Starting t-SNE with ${nPoints} points, perplexity=${perplexity}`);

        const distances = this.computePairwiseDistances(features);

        const P = this.computeAffinities(distances, perplexity);

        const Y = this.initializeEmbedding(nPoints, data, seed, random);

        const initialLearningRate = Math.min(200, nPoints / 4);
        const finalLearningRate = Math.min(50, nPoints / 12);
        const momentum = 0.5;
        const finalMomentum = 0.8;
        
        const velocities = Array(nPoints).fill(null).map(() => ({x: 0, y: 0}));

        const earlyExaggeration = 12.0;
        const stopExaggeration = Math.min(250, iterations / 2);

        for (let iter = 0; iter < iterations; iter++) {
            const progress = iter / iterations;
            const learningRate = initialLearningRate * (1 - progress) + finalLearningRate * progress;
            const currentMomentum = iter < stopExaggeration ? momentum : 
                momentum * (1 - progress) + finalMomentum * progress;
            
            const exaggeration = iter < stopExaggeration ? earlyExaggeration : 1.0;

            const {Q, Z} = this.computeLowDProbabilities(Y);

            const forces = this.computeTSNEForces(Y, P, Q, Z, exaggeration);

            this.clipGradients(forces, 50);

            for (let i = 0; i < nPoints; i++) {
                velocities[i].x = currentMomentum * velocities[i].x - learningRate * forces[i].x;
                velocities[i].y = currentMomentum * velocities[i].y - learningRate * forces[i].y;
                
                Y[i].x += velocities[i].x;
                Y[i].y += velocities[i].y;
            }

            if (iter % 50 === 0) {
                this.centerEmbedding(Y);
            }
        }

        console.log(`t-SNE completed. Final positions range: X[${Math.min(...Y.map(p => p.x)).toFixed(2)}, ${Math.max(...Y.map(p => p.x)).toFixed(2)}], Y[${Math.min(...Y.map(p => p.y)).toFixed(2)}, ${Math.max(...Y.map(p => p.y)).toFixed(2)}]`);
        
        return Y;
    }

    static computeTSNEForces(Y, P, Q, Z, exaggeration) {
        const nPoints = Y.length;
        const forces = Array(nPoints).fill(null).map(() => ({x: 0, y: 0}));

        for (let i = 0; i < nPoints; i++) {
            for (let j = 0; j < nPoints; j++) {
                if (i !== j) {
                    const dx = Y[i].x - Y[j].x;
                    const dy = Y[i].y - Y[j].y;
                    const dist2 = Math.max(1e-12, dx * dx + dy * dy);
                    
                    // t-SNE gradient: 4 * (P_ij - Q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)
                    const pij = exaggeration * P[i][j];
                    const qij = Q[i][j];
                    const mult = 4 * (pij - qij) / (1 + dist2);
                    
                    forces[i].x += mult * dx;
                    forces[i].y += mult * dy;
                }
            }
        }
        
        return forces;
    }

    static computeLowDProbabilities(Y) {
        const nPoints = Y.length;
        const Q = Array(nPoints).fill(null).map(() => Array(nPoints).fill(0));
        let Z = 0;

        // compute unnormalized Q and Z
        for (let i = 0; i < nPoints; i++) {
            for (let j = 0; j < nPoints; j++) {
                if (i !== j) {
                    const dx = Y[i].x - Y[j].x;
                    const dy = Y[i].y - Y[j].y;
                    const dist2 = dx * dx + dy * dy;
                    const qij = 1 / (1 + dist2); // t-distribution with 1 degree of freedom
                    Q[i][j] = qij;
                    Z += qij;
                }
            }
        }

        // norm
        Z = Math.max(Z, 1e-12);
        for (let i = 0; i < nPoints; i++) {
            for (let j = 0; j < nPoints; j++) {
                if (i !== j) {
                    Q[i][j] = Math.max(Q[i][j] / Z, 1e-12);
                }
            }
        }

        return {Q, Z};
    }

    static computeAffinities(distances, perplexity) {
        const n = distances.length;
        const P = Array(n).fill(null).map(() => Array(n).fill(0));
        const targetEntropy = Math.log2(perplexity);
        
        // find appropriate bandwidth (sigma)
        for (let i = 0; i < n; i++) {
            let sigma = 1.0;
            let betaMin = 0;
            let betaMax = Infinity;
            
            // binary search for the right sigma
            for (let tries = 0; tries < 50; tries++) {
                const beta = 1 / (2 * sigma * sigma);
                
                // conditional probabilities
                let sum = 0;
                const prob = Array(n).fill(0);
                for (let j = 0; j < n; j++) {
                    if (i !== j) {
                        prob[j] = Math.exp(-beta * distances[i][j] * distances[i][j]);
                        sum += prob[j];
                    }
                }
                
                if (sum === 0) {
                    sigma *= 2;
                    continue;
                }
                
                // norm and compute entropy
                let entropy = 0;
                for (let j = 0; j < n; j++) {
                    if (i !== j) {
                        prob[j] /= sum;
                        if (prob[j] > 1e-12) {
                            entropy -= prob[j] * Math.log2(prob[j]);
                        }
                        P[i][j] = prob[j];
                    }
                }
                
                // check if close enough to target entropy
                if (Math.abs(entropy - targetEntropy) < 1e-5) break;
                
                // adjust sigma
                if (entropy > targetEntropy) {
                    betaMax = beta;
                    sigma = (betaMin === 0) ? sigma / 2 : Math.sqrt(1 / ((betaMin + betaMax) / 2));
                } else {
                    betaMin = beta;
                    sigma = (betaMax === Infinity) ? sigma * 2 : Math.sqrt(1 / ((betaMin + betaMax) / 2));
                }
            }
        }
        
        // symmetrize: P_ij = (P_j|i + P_i|j) / (2*n)
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const pij = (P[i][j] + P[j][i]) / (2 * n);
                P[i][j] = Math.max(pij, 1e-12);
                P[j][i] = Math.max(pij, 1e-12);
            }
        }
        
        return P;
    }

    static initializeEmbedding(nPoints, data, seed, random) {
        const Y = [];
        const initStrategy = seed ? seed % 3 : 0;
        
        if (initStrategy === 0) {
            for (let i = 0; i < nPoints; i++) {
                Y.push({
                    x: (random() - 0.5) * 0.0001,
                    y: (random() - 0.5) * 0.0001,
                    id: data.points[i].id,
                    label: data.points[i].label,
                    originalData: data.points[i]
                });
            }
        } else if (initStrategy === 1) {
            for (let i = 0; i < nPoints; i++) {
                Y.push({
                    x: (random() - 0.5) * 2,
                    y: (random() - 0.5) * 2,
                    id: data.points[i].id,
                    label: data.points[i].label,
                    originalData: data.points[i]
                });
            }
        } else {
            for (let i = 0; i < nPoints; i++) {
                const angle = (i / nPoints) * 2 * Math.PI;
                Y.push({
                    x: Math.cos(angle) * 5 + (random() - 0.5) * 0.5,
                    y: Math.sin(angle) * 5 + (random() - 0.5) * 0.5,
                    id: data.points[i].id,
                    label: data.points[i].label,
                    originalData: data.points[i]
                });
            }
        }
        
        return Y;
    }

    static clipGradients(forces, maxNorm) {
        for (let i = 0; i < forces.length; i++) {
            const norm = Math.sqrt(forces[i].x * forces[i].x + forces[i].y * forces[i].y);
            if (norm > maxNorm) {
                forces[i].x = (forces[i].x / norm) * maxNorm;
                forces[i].y = (forces[i].y / norm) * maxNorm;
            }
        }
    }

    static centerEmbedding(Y) {
        const meanX = Y.reduce((sum, p) => sum + p.x, 0) / Y.length;
        const meanY = Y.reduce((sum, p) => sum + p.y, 0) / Y.length;
        Y.forEach(p => {
            p.x -= meanX;
            p.y -= meanY;
        });
    }

    static fastUMAP(data, nNeighbors = 15, minDist = 0.1, iterations = 200, seed = null) {
        const random = seed !== null ? mulberry32(seed) : Math.random;
        const nPoints = data.points.length;
        
        // Reduce dataset size if too large for web performance
        const maxPoints = 150;
        let processedData = data;
        if (nPoints > maxPoints) {
            console.log(`Reducing UMAP dataset from ${nPoints} to ${maxPoints} points for performance`);
            processedData = this.subsampleData(data, maxPoints);
        }
        
        const actualNPoints = processedData.points.length;
        const features = processedData.points.map(p => p.features);
    
        console.log(`Starting UMAP with ${actualNPoints} points, n_neighbors=${Math.min(nNeighbors, actualNPoints-1)}`);

        const maxIterations = Math.min(iterations, 150);

        const distances = this.computeFastPairwiseDistances(features);
        const actualNeighbors = Math.min(nNeighbors, actualNPoints - 1);
        const knnGraph = this.buildKNNGraph(distances, actualNeighbors);

        const Y = this.betterSpectralInit(features, actualNPoints, random);
        
        Y.forEach((pos, i) => {
            pos.id = processedData.points[i].id;
            pos.label = processedData.points[i].label;
            pos.originalData = processedData.points[i];
        });

        const a = 1.929;
        const b = 0.7915;
        const initialLearningRate = 1.0;
        const finalLearningRate = 0.1;
        
        const startTime = Date.now();
        const maxTime = 8000; // 8 second timeout
        
        for (let iter = 0; iter < maxIterations; iter++) {
            if (Date.now() - startTime > maxTime) {
                console.log('UMAP timeout - stopping early');
                break;
            }
            
            const progress = iter / maxIterations;
            const learningRate = initialLearningRate * (1 - progress) + finalLearningRate * progress;
    
            // attractive forces w/ bounds checking
            for (let i = 0; i < actualNPoints; i++) {
                const neighbors = knnGraph[i] || [];
                for (let j of neighbors) {
                    if (j >= actualNPoints) continue;
                    
                    const dx = Y[i].x - Y[j].x;
                    const dy = Y[i].y - Y[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy + 1e-10);
                    
                    const attraction = this.umapAttractionGrad(dist, a, b);
                    
                    const fx = attraction * dx / dist;
                    const fy = attraction * dy / dist;
                    
                    // clip forces to prevent explosion
                    const clippedFx = Math.max(-10, Math.min(10, fx));
                    const clippedFy = Math.max(-10, Math.min(10, fy));
                    
                    Y[i].x -= learningRate * clippedFx;
                    Y[i].y -= learningRate * clippedFy;
                    Y[j].x += learningRate * clippedFx;
                    Y[j].y += learningRate * clippedFy;
                }
            }
    
            // few negative samples for performance
            const negativeSamples = Math.min(3, actualNPoints - 1);
            for (let i = 0; i < actualNPoints; i++) {
                for (let s = 0; s < negativeSamples; s++) {
                    let j = Math.floor(random() * actualNPoints);
                    let attempts = 0;
                    while ((j === i || (knnGraph[i] && knnGraph[i].includes(j))) && attempts < 10) {
                        j = Math.floor(random() * actualNPoints);
                        attempts++;
                    }
                    if (attempts >= 10) continue; // skip if no good neg sample
                    
                    const dx = Y[i].x - Y[j].x;
                    const dy = Y[i].y - Y[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy + 1e-10);
                    
                    const repulsion = this.umapRepulsionGrad(dist, a, b, minDist);
                    
                    if (repulsion > 0) {
                        const fx = repulsion * dx / dist;
                        const fy = repulsion * dy / dist;
                        
                        // Clip forces
                        const clippedFx = Math.max(-5, Math.min(5, fx));
                        const clippedFy = Math.max(-5, Math.min(5, fy));
                        
                        Y[i].x += learningRate * clippedFx;
                        Y[i].y += learningRate * clippedFy;
                    }
                }
            }

            for (let i = 0; i < actualNPoints; i++) {
                Y[i].x = Math.max(-200, Math.min(200, Y[i].x));
                Y[i].y = Math.max(-200, Math.min(200, Y[i].y));
            }
        }
        
        console.log(`UMAP completed in ${Date.now() - startTime}ms. Final positions range: X[${Math.min(...Y.map(p => p.x)).toFixed(2)}, ${Math.max(...Y.map(p => p.x)).toFixed(2)}], Y[${Math.min(...Y.map(p => p.y)).toFixed(2)}, ${Math.max(...Y.map(p => p.y)).toFixed(2)}]`);
        
        return Y;
    }

    static subsampleData(data, targetSize) {
        // keep representative points from each cluster
        const clusters = {};
        data.points.forEach(point => {
            const cluster = point.cluster || 0;
            if (!clusters[cluster]) clusters[cluster] = [];
            clusters[cluster].push(point);
        });
        
        const result = [];
        const pointsPerCluster = Math.floor(targetSize / Object.keys(clusters).length);
        
        Object.values(clusters).forEach(clusterPoints => {
            // Randomly sample from each cluster
            const shuffled = clusterPoints.sort(() => Math.random() - 0.5);
            result.push(...shuffled.slice(0, pointsPerCluster));
        });
        
        // Fill remaining spots randomly
        while (result.length < targetSize && result.length < data.points.length) {
            const remainingPoints = data.points.filter(p => !result.includes(p));
            if (remainingPoints.length === 0) break;
            result.push(remainingPoints[Math.floor(Math.random() * remainingPoints.length)]);
        }
        
        return {
            points: result.slice(0, targetSize),
            metadata: data.metadata
        };
    }

    static computeFastPairwiseDistances(features) {
        const n = features.length;
        const distances = Array(n).fill(null).map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                let sum = 0;
                // fewer dimensions for speed (first 15 instead of all)
                const dims = Math.min(features[i].length, 15);
                for (let d = 0; d < dims; d++) {
                    sum += Math.pow(features[i][d] - features[j][d], 2);
                }
                const dist = Math.sqrt(sum);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        return distances;
    }

    static umapAttractionGrad(dist, a, b) {
        return -2 * a * b * Math.pow(dist, b - 1) / (1 + a * Math.pow(dist, 2 * b));
    }

    static umapRepulsionGrad(dist, a, b, minDist) {
        return 2 * b / ((minDist + dist) * (1 + a * Math.pow(dist, 2 * b)));
    }

    static betterSpectralInit(features, nPoints, random) {
        const positions = [];
        
        if (features.length > 0 && features[0].length >= 2) {
            // Use first two dimensions as initial approximation, but add noise
            for (let i = 0; i < nPoints; i++) {
                positions.push({
                    x: features[i][0] * 10 + (random() - 0.5) * 2,
                    y: features[i][1] * 10 + (random() - 0.5) * 2
                });
            }
        } else {
            // Fallback to random circular arrangement
            for (let i = 0; i < nPoints; i++) {
                const angle = (i / nPoints) * 2 * Math.PI;
                positions.push({
                    x: Math.cos(angle) * 10 + (random() - 0.5) * 2,
                    y: Math.sin(angle) * 10 + (random() - 0.5) * 2
                });
            }
        }
        
        return positions;
    }

    static computePairwiseDistances(features) {
        const n = features.length;
        const distances = Array(n).fill(null).map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                let sum = 0;
                // Use all dimensions, not just first 10
                for (let d = 0; d < features[i].length; d++) {
                    sum += Math.pow(features[i][d] - features[j][d], 2);
                }
                const dist = Math.sqrt(sum);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        return distances;
    }

    static buildKNNGraph(distances, k) {
        const n = distances.length;
        const knnGraph = Array(n).fill(null).map(() => []);
        
        for (let i = 0; i < n; i++) {
            const dists = distances[i]
                .map((d, j) => ({dist: d, idx: j}))
                .filter(item => item.idx !== i);
            
            dists.sort((a, b) => a.dist - b.dist);
            
            for (let j = 0; j < Math.min(k, dists.length); j++) {
                knnGraph[i].push(dists[j].idx);
            }
        }
        
        return knnGraph;
    }
}

class InteractiveEmbeddingExperiments {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.dataGenerator = new DataGenerator();
        this.currentData = null;
        this.currentEmbedding = null;
        this.visualizations = new Map();
    }

    initExperiment(experimentType) {
        switch(experimentType) {
            case 'random-patterns':
                this.initRandomPatternsExperiment();
                break;
            case 'parameter-manipulation':
                this.initParameterManipulation();
                break;
            case 'cherry-picking':
                this.initCherryPicking();
                break;
            case 'distance-deception':
                this.initDistanceDeception();
                break;
            case 'scale-manipulation':
                this.initScaleManipulation();
                break;
        }
    }

    initRandomPatternsExperiment() {
        const container = this.container;
        if (!container) return;

        this.currentData = this.dataGenerator.generateRandomData(80, 20);

        const controls = document.createElement('div');
        controls.className = 'experiment-controls';
        controls.innerHTML = `
            <div class="control-row">
                <label for="exp1-algorithm">Algorithm:</label>
                <select id="exp1-algorithm">
                    <option value="tsne">t-SNE</option>
                    <option value="umap">UMAP</option>
                </select>
            </div>
            <div class="control-row">
                <label for="exp1-perplexity">Perplexity / n_neighbors:</label>
                <input type="range" id="exp1-perplexity" min="5" max="50" value="15" step="5">
                <span id="exp1-perplexity-value">15</span>
            </div>
            <div class="control-row">
                <button id="exp1-new-data">Generate New Random Data</button>
                <button id="exp1-rerun">Re-run with Different Seed</button>
            </div>
        `;
        
        container.appendChild(controls);

        const vizContainer = document.createElement('div');
        vizContainer.className = 'viz-grid';
        vizContainer.innerHTML = `
            <div class="viz-panel">
                <h4>Original Data (First 2 of 50 Random Dimensions)</h4>
                <div id="exp1-original" class="plot-area"></div>
                <p style="font-size: 12px; color: #666; margin: 10px 0 0 0;">
                    This is truly random data - no patterns, no clusters, just noise.
                </p>
            </div>
            <div class="viz-panel">
                <h4>After Dimensionality Reduction - Fake Patterns!</h4>
                <div id="exp1-embedded" class="plot-area" style="max-width: 100%; overflow: hidden;></div>
                <p style="font-size: 12px; color: #666; margin: 10px 0 0 0;">
                    Look! "Clusters" appear where none exist. The algorithm creates structure from randomness.
                </p>
            </div>
        `;
        
        container.appendChild(vizContainer);

        const insights = document.createElement('div');
        insights.className = 'insight-box';
        insights.innerHTML = `
            <h4>Apophenia!</h4>
            <p><strong>What's happening:</strong> You're looking at completely random data - like TV static. 
            But after dimensionality reduction, you see "clusters" and "patterns". They're not real!</p>
            <p>McNutt et al. (2020) found that <strong>over 80% of people</strong> confidently identify 
            meaningful patterns in visualizations of random data. Our brains are wired to find patterns, 
            even when they don't exist.</p>
        `;
        
        container.appendChild(insights);
        this.setupExperiment1Listeners();
        this.updateExperiment1();
    }

    setupExperiment1Listeners() {
        document.getElementById('exp1-perplexity').addEventListener('input', (e) => {
            document.getElementById('exp1-perplexity-value').textContent = e.target.value;
            this.updateExperiment1();
        });
        
        document.getElementById('exp1-algorithm').addEventListener('change', () => {
            this.updateExperiment1();
        });
        
        document.getElementById('exp1-new-data').addEventListener('click', () => {
            this.currentData = this.dataGenerator.generateRandomData(500, 50);
            this.updateExperiment1();
        });
        
        document.getElementById('exp1-rerun').addEventListener('click', () => {
            this.updateExperiment1();
        });
    }

    updateExperiment1() {
        const algorithm = document.getElementById('exp1-algorithm').value;
        const perplexity = parseInt(document.getElementById('exp1-perplexity').value);
        
        // show first 2 dimensions of original data
        const originalPoints = this.currentData.points.map(p => ({
            x: p.features[0],
            y: p.features[1],
            id: p.id,
            label: p.label
        }));
        
        this.createScatterPlot('exp1-original', originalPoints, {
            title: 'Random Data',
            showClusters: false
        });

        if (algorithm === 'tsne') {
            this.currentEmbedding = EmbeddingAlgorithms.tSNE(
                this.currentData, 
                perplexity,
                1500,
                Math.random() * 1000
            );
        } else {
            this.currentEmbedding = EmbeddingAlgorithms.fastUMAP(
                this.currentData,
                perplexity,
                0.1,
                1500,
                Math.random() * 1000
            );
        }
        
        this.createScatterPlot('exp1-embedded', this.currentEmbedding, {
            title: `${algorithm.toUpperCase()} Result`,
            showClusters: true,
            colorByDistance: true
        });
    }

    initParameterManipulation() {
        const container = this.container;
        if (!container) return;

        this.currentData = this.dataGenerator.generateClusteredData(200, 50, 4);
        
        const controls = document.createElement('div');
        controls.className = 'experiment-controls';
        controls.innerHTML = `
            <div class="control-row">
                <label for="exp2-algorithm">Algorithm:</label>
                <select id="exp2-algorithm">
                    <option value="tsne">t-SNE</option>
                    <option value="umap">UMAP</option>
                </select>
            </div>
            <div class="control-row">
                <label for="exp2-perplexity">Perplexity / n_neighbors:</label>
                <input type="range" id="exp2-perplexity" min="2" max="50" value="30" step="1">
                <span id="exp2-perplexity-value">30</span>
            </div>
            <div class="control-row">
                <label for="exp2-min-dist">Min Distance (works for UMAP only):</label>
                <input type="range" id="exp2-min-dist" min="0.01" max="1.0" value="0.1" step="0.01">
                <span id="exp2-min-dist-value">0.1</span>
            </div>
            <div class="parameter-info">
                <p><strong>Different parameters can tell different stories:</strong></p>
                <ul>
                    <li>Very low perplexity: Creates tight, separated clusters</li>
                    <li>Very high perplexity: Merges clusters together</li>
                    <li>Small min_dist: Points cluster tightly</li>
                    <li>Large min_dist: Points spread out</li>
                </ul>
            </div>
        `;
        
        container.appendChild(controls);
        
        const vizContainer = document.createElement('div');
        vizContainer.className = 'viz-container single';
        vizContainer.innerHTML = `
            <div class="viz-panel large">
                <h4>Same Data, Different Parameters, Different "Truth"</h4>
                <div id="exp2-plot" class="plot-area"></div>
            </div>
        `;
        
        container.appendChild(vizContainer);
        
        const insights = document.createElement('div');
        insights.className = 'insight-box';
        insights.innerHTML = `
            <h4>Note</h4>
            <p>Wattenberg et al. (2016) showed that t-SNE is so sensitive to hyperparameters that 
            you can make almost any dataset look like it has distinct clusters OR continuous structure. 
            A researcher could try many parameter settings and cherry-pick the one that "looks right".</p>
        `;
        
        container.appendChild(insights);
        
        this.setupExperiment2Listeners();
        this.updateExperiment2();
    }

    setupExperiment2Listeners() {
        const update = () => this.updateExperiment2();
        
        document.getElementById('exp2-perplexity').addEventListener('input', (e) => {
            document.getElementById('exp2-perplexity-value').textContent = e.target.value;
            update();
        });
        
        document.getElementById('exp2-min-dist').addEventListener('input', (e) => {
            document.getElementById('exp2-min-dist-value').textContent = e.target.value;
            update();
        });
        
        document.getElementById('exp2-algorithm').addEventListener('change', update);
    }

    updateExperiment2() {
        const algorithm = document.getElementById('exp2-algorithm').value;
        const perplexity = parseInt(document.getElementById('exp2-perplexity').value);
        const minDist = parseFloat(document.getElementById('exp2-min-dist').value);
        
        if (algorithm === 'tsne') {
            this.currentEmbedding = EmbeddingAlgorithms.tSNE(
                this.currentData, 
                perplexity,
                1500,
                42
            );
            console.log('EXPERIMENT2 Result length:', this.currentEmbedding.length);
            console.log('EXPERIMENT2 First few results:', this.currentEmbedding.slice(0, 5));
            console.log('EXPERIMENT2 Unique X values:', new Set(this.currentEmbedding.map(p => p[0])).size);
            console.log('EXPERIMENT2 Unique Y values:', new Set(this.currentEmbedding.map(p => p[1])).size);
        } else {
            this.currentEmbedding = EmbeddingAlgorithms.fastUMAP(
                this.currentData,
                perplexity,
                minDist,
                1500,
                42
            );
        }
        
        this.createScatterPlot('exp2-plot', this.currentEmbedding, {
            title: `${algorithm.toUpperCase()} with perplexity=${perplexity}${algorithm === 'umap' ? `, min_dist=${minDist}` : ''}`,
            colorByCluster: true
        });
    }

    initCherryPicking() {
        const container = this.container;
        if (!container) return;

        this.currentData = this.dataGenerator.generateClusteredData(200, 50, 3);
        this.cherryPickingResults = [];
        
        const controls = document.createElement('div');
        controls.className = 'experiment-controls';
        controls.innerHTML = `
            <div class="control-row">
                <button id="exp3-run-10">Run 10 Different Seeds</button>
                <button id="exp3-show-best" disabled>Show "Best" Result</button>
                <button id="exp3-show-worst" disabled>Show "Worst" Result</button>
                <button id="exp3-show-all" disabled>Show All Results</button>
            </div>
            <div id="exp3-status" class="status-message"></div>
        `;
        
        container.appendChild(controls);
        
        const vizContainer = document.createElement('div');
        vizContainer.className = 'viz-grid';
        vizContainer.innerHTML = `
            <div id="exp3-current-viz" class="viz-panel">
                <h4>Current Result</h4>
                <div id="exp3-plot" class="plot-area"></div>
            </div>
            <div id="exp3-all-viz" class="viz-panel" style="display:none;">
                <h4>All 10 Results</h4>
                <div id="exp3-grid" class="mini-plots-grid"></div>
            </div>
        `;
        
        container.appendChild(vizContainer);
        
        // const insights = document.createElement('div');
        // insights.className = 'insight-box warning';
        // insights.innerHTML = `
        //     <h4>Visual P-Hacking</h4>
        // `;
        
        // container.appendChild(insights);
        
        this.setupExperiment3Listeners();
    }

    setupExperiment3Listeners() {
        document.getElementById('exp3-run-10').addEventListener('click', () => {
            this.runMultipleEmbeddings();
        });
        
        document.getElementById('exp3-show-best').addEventListener('click', () => {
            this.showCherryPickedResult('best');
        });
        
        document.getElementById('exp3-show-worst').addEventListener('click', () => {
            this.showCherryPickedResult('worst');
        });
        
        document.getElementById('exp3-show-all').addEventListener('click', () => {
            this.showAllResults();
        });
    }

    // metric that captures what makes a viz "compelling" for storytelling
    calculateStorytellingQuality(embedding) {
        const clusters = {};
        
        embedding.forEach(point => {
            const cluster = point.originalData.cluster || 0;
            if (!clusters[cluster]) {
                clusters[cluster] = { points: [], x: 0, y: 0 };
            }
            clusters[cluster].points.push(point);
            clusters[cluster].x += point.x;
            clusters[cluster].y += point.y;
        });

        Object.values(clusters).forEach(c => {
            c.x /= c.points.length;
            c.y /= c.points.length;
        });
        
        // "visual convincingness": how separated do the clusters look?
        const clusterArray = Object.values(clusters);
        
        // 1. dist between cluster visual centers
        let totalBetweenDist = 0;
        let pairs = 0;
        for (let i = 0; i < clusterArray.length; i++) {
            for (let j = i + 1; j < clusterArray.length; j++) {
                totalBetweenDist += Math.sqrt(
                    Math.pow(clusterArray[i].x - clusterArray[j].x, 2) +
                    Math.pow(clusterArray[i].y - clusterArray[j].y, 2)
                );
                pairs++;
            }
        }
        const avgBetweenDist = pairs > 0 ? totalBetweenDist / pairs : 0;
        
        // 2. tightness of individual clusters
        let totalWithinVariance = 0;
        Object.values(clusters).forEach(cluster => {
            let variance = 0;
            cluster.points.forEach(point => {
                variance += Math.pow(point.x - cluster.x, 2) + Math.pow(point.y - cluster.y, 2);
            });
            totalWithinVariance += variance / cluster.points.length;
        });
        const avgWithinVariance = totalWithinVariance / clusterArray.length;
        
        // 3. "aesthetic appeal" - not too spread out, not too cramped
        const allX = embedding.map(p => p.x);
        const allY = embedding.map(p => p.y);
        const spread = (Math.max(...allX) - Math.min(...allX)) + (Math.max(...allY) - Math.min(...allY));
        const aestheticPenalty = Math.abs(spread - 40) / 40;
        
        // higher score = more "convincing" visualization
        return avgBetweenDist / Math.sqrt(avgWithinVariance + 1) - aestheticPenalty;
    }

    runMultipleEmbeddings() {
        const status = document.getElementById('exp3-status');
        status.textContent = 'Running 10 different embeddings with realistic parameters...';
        
        this.cherryPickingResults = [];

        document.getElementById('exp3-plot').innerHTML = '';
        document.getElementById('exp3-grid').innerHTML = '';

        // fresh clustered data but more ambiguous (less obvious clusters)
        this.currentData = this.dataGenerator.generateAmbiguousData(180, 40, 3);

        ['exp3-show-best', 'exp3-show-worst', 'exp3-show-all'].forEach(id => {
            document.getElementById(id).disabled = true;
        });
        
        const results = [];
        let completed = 0;

        // parameter combos!
        const realisticCombinations = [
            // t-SNE with different seeds (most common cherry-picking)
            {algorithm: 'tsne', perplexity: 30, seed: 42, description: 'Standard (seed=42)'},
            {algorithm: 'tsne', perplexity: 30, seed: 123, description: 'Standard (seed=123)'},
            {algorithm: 'tsne', perplexity: 30, seed: 999, description: 'Standard (seed=999)'},
            
            // perplexity range exploration
            {algorithm: 'tsne', perplexity: 10, seed: 42, description: 'Lower perplexity'},
            {algorithm: 'tsne', perplexity: 50, seed: 42, description: 'Higher perplexity'},
            {algorithm: 'tsne', perplexity: 15, seed: 42, description: 'Conservative perplexity'},
            
            // algo shopping (t-SNE vs UMAP)
            {algorithm: 'umap', nNeighbors: 15, minDist: 0.1, seed: 42, description: 'UMAP standard'},
            {algorithm: 'umap', nNeighbors: 30, minDist: 0.1, seed: 42, description: 'UMAP higher neighbors'},
            {algorithm: 'umap', nNeighbors: 15, minDist: 0.5, seed: 42, description: 'UMAP spread out'},

            {algorithm: 'tsne', perplexity: 25, seed: 777, description: 'Alternative params'}
        ];

        const runNextEmbedding = () => {
            if (completed < 10) {
                status.textContent = `Running ${realisticCombinations[completed].description}...`;

                setTimeout(() => {
                    const params = realisticCombinations[completed];
                    let embedding;
                    
                    if (params.algorithm === 'tsne') {
                        embedding = EmbeddingAlgorithms.tSNE(
                            this.currentData,
                            params.perplexity,
                            1000,
                            params.seed
                        );
                    } else {
                        embedding = EmbeddingAlgorithms.fastUMAP(
                            this.currentData,
                            params.nNeighbors || 15,
                            params.minDist || 0.1,
                            300,
                            params.seed
                        );
                    }

                    // "story-telling" appeal
                    const quality = this.calculateStorytellingQuality(embedding);

                    results.push({
                        embedding,
                        quality,
                        params,
                        description: params.description
                    });

                    completed++;
                    runNextEmbedding();
                }, 200);
            } else {
                this.cherryPickingResults = results.sort((a, b) => b.quality - a.quality);

                const bestDesc = this.cherryPickingResults[0].description;
                const worstDesc = this.cherryPickingResults[this.cherryPickingResults.length-1].description;
                
                status.textContent = `Done! Best: "${bestDesc}" vs Worst: "${worstDesc}"`;

                ['exp3-show-best', 'exp3-show-worst', 'exp3-show-all'].forEach(id => {
                    document.getElementById(id).disabled = false;
                });

                this.createScatterPlot('exp3-plot', this.cherryPickingResults[0].embedding, {
                    title: `${bestDesc} - "Clearest" clusters`,
                    colorByCluster: true
                });
            }
        };

        runNextEmbedding();
    }

    calculateClusterSeparation(embedding) {
        // avg distance between cluster centers
        const clusters = {};
        
        embedding.forEach(point => {
            const cluster = point.originalData.cluster || 0;
            if (!clusters[cluster]) {
                clusters[cluster] = { x: 0, y: 0, count: 0 };
            }
            clusters[cluster].x += point.x;
            clusters[cluster].y += point.y;
            clusters[cluster].count++;
        });

        Object.values(clusters).forEach(c => {
            c.x /= c.count;
            c.y /= c.count;
        });
        
        // ag distance between centroids
        const clusterArray = Object.values(clusters);
        let totalDist = 0;
        let pairs = 0;
        
        for (let i = 0; i < clusterArray.length; i++) {
            for (let j = i + 1; j < clusterArray.length; j++) {
                const dist = Math.sqrt(
                    Math.pow(clusterArray[i].x - clusterArray[j].x, 2) +
                    Math.pow(clusterArray[i].y - clusterArray[j].y, 2)
                );
                totalDist += dist;
                pairs++;
            }
        }
        
        return pairs > 0 ? totalDist / pairs : 0;
    }

    showCherryPickedResult(type) {
        const result = type === 'best' 
            ? this.cherryPickingResults[0] 
            : this.cherryPickingResults[this.cherryPickingResults.length - 1];
        
        this.createScatterPlot('exp3-plot', result.embedding, {
            title: `${type === 'best' ? 'Best' : 'Worst'} Result (${type === 'best' ? 'Most' : 'Least'} Separated Clusters)`,
            colorByCluster: true
        });
        
        document.getElementById('exp3-all-viz').style.display = 'none';
        document.getElementById('exp3-current-viz').style.display = 'block';
    }

    showAllResults() {
        document.getElementById('exp3-current-viz').style.display = 'none';
        document.getElementById('exp3-all-viz').style.display = 'block';
        
        const grid = document.getElementById('exp3-grid');
        grid.innerHTML = '';

        const containers = [];
        this.cherryPickingResults.forEach((result, i) => {
            const miniContainer = document.createElement('div');
            miniContainer.className = 'mini-plot';
            miniContainer.id = `exp3-mini-container-${i}`;
            
            // inner div for plot
            const plotDiv = document.createElement('div');
            plotDiv.id = `exp3-mini-${i}`;
            plotDiv.style.width = '100%';
            plotDiv.style.height = '100%';
            miniContainer.appendChild(plotDiv);
            
            grid.appendChild(miniContainer);
            containers.push({id: `exp3-mini-${i}`, result: result, index: i});
        });
        
        // layout reflow before creating plots
        grid.offsetHeight;
        
        // create plots after DOM has updated
        setTimeout(() => {
            containers.forEach(({id, result, index}) => {
                this.createMiniScatterPlot(id, result.embedding, {
                    title: `#${index + 1}`,
                    colorByCluster: true
                });
            });
        }, 100);
    }

    createMiniScatterPlot(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        container.innerHTML = '';

        const parentContainer = container.parentElement;
        const parentRect = parentContainer.getBoundingClientRect();

        const size = Math.min(parentRect.width - 10, parentRect.height - 10, 120);
        
        const margin = {top: 20, right: 5, bottom: 5, left: 5};
        const width = size - margin.left - margin.right;
        const height = size - margin.top - margin.bottom;

        const svg = d3.select(`#${containerId}`)
            .append('svg')
            .attr('width', size)
            .attr('height', size)
            .attr('viewBox', `0 0 ${size} ${size}`)
            .style('display', 'block')
            .style('margin', 'auto');
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        g.append('text')
            .attr('x', width / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .style('font-size', '10px')
            .style('font-weight', 'bold')
            .text(options.title || '');

        if (!data || data.length === 0) {
            g.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .style('font-size', '10px')
                .text('No data');
            return;
        }

        const xExtent = d3.extent(data, d => d.x);
        const yExtent = d3.extent(data, d => d.y);

        const xPadding = (xExtent[1] - xExtent[0]) * 0.15;
        const yPadding = (yExtent[1] - yExtent[0]) * 0.15;
        
        const xScale = d3.scaleLinear()
            .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
            .range([0, width]);
        
        const yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
            .range([height, 0]);
        
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        g.selectAll('.mini-point')
            .data(data)
            .enter().append('circle')
            .attr('class', 'mini-point')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 1.2)
            .attr('fill', (d) => {
                if (options.colorByCluster && d.originalData && d.originalData.cluster !== undefined) {
                    return colorScale(d.originalData.cluster);
                }
                return colorScale(0);
            })
            .attr('opacity', 0.8)
            .attr('stroke', 'none');
    }

    createScatterPlot(containerId, data, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';

        const containerRect = container.getBoundingClientRect();
        const containerWidth = containerRect.width - 20;
        const containerHeight = Math.max(300, containerRect.height - 20);

        const margin = {top: 30, right: 30, bottom: 40, left: 40};
        const width = containerWidth - margin.left - margin.right;
        const height = containerHeight - margin.top - margin.bottom;
        
        const svg = d3.select(`#${containerId}`)
            .append('svg')
            .attr('width', containerWidth)
            .attr('height', containerHeight)
            .style('display', 'block');
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const xExtent = d3.extent(data, d => d.x);
        const yExtent = d3.extent(data, d => d.y);
        const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
        const yPadding = (yExtent[1] - yExtent[0]) * 0.1;
        
        const xScale = d3.scaleLinear()
            .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
            .range([0, width]);
        
        const yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
            .range([height, 0]);

        let colorScale;
        if (options.colorByDistance) {
            const centerX = d3.mean(data, d => d.x);
            const centerY = d3.mean(data, d => d.y);
            const distances = data.map(d => 
                Math.sqrt(Math.pow(d.x - centerX, 2) + Math.pow(d.y - centerY, 2))
            );
            colorScale = d3.scaleSequential(d3.interpolateViridis)
                .domain([0, d3.max(distances)]);
        } else {
            colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        }

        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale).ticks(5));
        
        g.append('g')
            .call(d3.axisLeft(yScale).ticks(5));

        if (options.title) {
            g.append('text')
                .attr('x', width / 2)
                .attr('y', -10)
                .attr('text-anchor', 'middle')
                .style('font-size', '14px')
                .style('font-weight', 'bold')
                .text(options.title);
        }

        const points = g.selectAll('.point')
            .data(data)
            .enter().append('circle')
            .attr('class', 'point')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 3)
            .attr('fill', (d, i) => {
                if (options.colorByDistance) {
                    const centerX = d3.mean(data, d => d.x);
                    const centerY = d3.mean(data, d => d.y);
                    const distance = Math.sqrt(
                        Math.pow(d.x - centerX, 2) + Math.pow(d.y - centerY, 2)
                    );
                    return colorScale(distance);
                } else if (options.colorByCluster && d.originalData && d.originalData.cluster !== undefined) {
                    return colorScale(d.originalData.cluster);
                } else if (options.colorByGroup && d.originalData && d.originalData.hiddenGroup !== undefined) {
                    return colorScale(d.originalData.hiddenGroup);
                } else {
                    return colorScale(i % 10);
                }
            })
            .attr('opacity', 0.7)
            .attr('stroke', 'white')
            .attr('stroke-width', 0.5);

        const tooltip = d3.select('body').selectAll('.tooltip').data([0])
            .join('div')
            .attr('class', 'tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none');
        
        points.on('mouseover', function(event, d) {
            d3.select(this).attr('r', 5);
            tooltip.transition().duration(200).style('opacity', .9);
            tooltip.html(`Point ${d.id}`)
                .style('left', (event.pageX + 5) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 3);
            tooltip.transition().duration(500).style('opacity', 0);
        });
        
        return {svg, xScale, yScale, points};
    }
}

window.InteractiveEmbeddingExperiments = InteractiveEmbeddingExperiments;
window.DataGenerator = DataGenerator;
window.EmbeddingAlgorithms = EmbeddingAlgorithms;
