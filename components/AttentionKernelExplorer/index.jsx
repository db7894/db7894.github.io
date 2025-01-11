import '../styles/main.css';
import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Slider } from '../ui/slider';

const gaussianKernel = (u, v, bandwidth) => {
  const diff = u - v;
  return Math.exp(-diff * diff / (2 * bandwidth * bandwidth));
};

const matrixMultiply = (matrix, vector) => {
  return matrix.map(row => 
    row.reduce((sum, val, i) => sum + val * vector[i], 0)
  );
};

const dotProduct = (v1, v2) => v1.reduce((sum, val, i) => sum + val * v2[i], 0);

const sum = arr => arr.reduce((a, b) => a + b, 0);

const generateInitialData = () => {
  const points = [];
  for (let i = 0; i < 50; i++) {
    const x = i / 50;
    const value = Math.sin(x * 4 * Math.PI) * Math.exp(-x) + Math.sin(x * 2 * Math.PI) * 0.5;
    
    // position encodings
    const pos = [
      Math.sin(x * 2 * Math.PI),
      Math.cos(x * 2 * Math.PI),
      x,
      1
    ];
    
    points.push({ x, value, pos });
  }
  return points;
};

const AttentionKernelExplorer = () => {
  const DIM = 4;
  const initializeWeights = () => {
    const w = Array(DIM).fill(0).map(() => Array(DIM).fill(0));
    // make attention focus on periodic similarity
    w[0][0] = 1.0;  // attention to first sinusoidal component
    w[1][1] = 1.0;  // and second sinusoidal component
    return w;
  };
  
  const [Wq, setWq] = useState(initializeWeights);
  const [Wk, setWk] = useState(initializeWeights);
  const [kernelBandwidth, setKernelBandwidth] = useState(0.1);

  const [temperature, setTemperature] = useState(1.0);
  
  const data = useMemo(() => generateInitialData(), []);

  const computeResults = useMemo(() => {
    return data.map((queryPoint, qIdx) => {
      const kernelWeights = data.map(point => 
        gaussianKernel(queryPoint.x, point.x, kernelBandwidth)
      );
      const kernelTotal = sum(kernelWeights);
      const normalizedKernelWeights = kernelWeights.map(w => w / kernelTotal);
      const kernelResult = data.reduce((sum, point, j) => 
        sum + normalizedKernelWeights[j] * point.value, 
      0);

      // attention w/ temperature scaling
      const query = matrixMultiply(Wq, queryPoint.pos);
      const attentionScores = data.map(keyPoint => {
        const key = matrixMultiply(Wk, keyPoint.pos);
        return Math.exp(dotProduct(query, key) / (Math.sqrt(DIM) * temperature));
      });
      
      const attentionTotal = sum(attentionScores);
      const attentionWeights = attentionScores.map(s => s / attentionTotal);
      const attentionResult = data.reduce((sum, point, j) => 
        sum + attentionWeights[j] * point.value,
      0);

      // weights for middle position to visualize attention patterns
      const weightProfile = qIdx === Math.floor(data.length/2) ? {
        kernelWeights: normalizedKernelWeights,
        attentionWeights: attentionWeights
      } : null;

      return {
        x: queryPoint.x,
        originalValue: queryPoint.value,
        kernel: kernelResult,
        attention: attentionResult,
        ...(weightProfile ? { 
          kernelWeight: normalizedKernelWeights,
          attentionWeight: attentionWeights 
        } : {})
      };
    });
  }, [data, Wq, Wk, kernelBandwidth, temperature]);

  // weight profiles for visualization
  const midpoint = Math.floor(data.length/2);
  const weightProfiles = {
    kernel: data.map((_, i) => ({
      x: data[i].x,
      weight: computeResults[midpoint]?.kernelWeight?.[i] || 0
    })),
    attention: data.map((_, i) => ({
      x: data[i].x,
      weight: computeResults[midpoint]?.attentionWeight?.[i] || 0
    }))
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Kernel Smoothing vs Attention</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Kernel Bandwidth: {kernelBandwidth.toFixed(3)}
              </label>
              <Slider 
                value={[kernelBandwidth]}
                onValueChange={([v]) => setKernelBandwidth(v)}
                min={0.01}
                max={0.5}
                step={0.01}
              />
              <div className="text-xs text-gray-500 mt-1">
                Controls locality of kernel smoothing
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Attention Temperature: {temperature.toFixed(2)}
              </label>
              <Slider 
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0.1}
                max={2.0}
                step={0.1}
              />
              <div className="text-xs text-gray-500 mt-1">
                Controls sharpness of attention focus
              </div>
            </div>
          </div>

          <div className="grid grid-rows-2 gap-4">
            <div className="h-64">
              <p className="text-sm font-medium mb-2">Values & Smoothing</p>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={computeResults}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" />
                  <YAxis domain={[-2, 2]} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="originalValue" 
                    stroke="#8884d8" 
                    name="Original Values"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="kernel" 
                    stroke="#ff7300" 
                    name="Kernel Smoothing"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="attention" 
                    stroke="#0088fe" 
                    name="Attention"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="h-64">
              <p className="text-sm font-medium mb-2">Weight Profiles at Input Sequence Middle Position</p>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={weightProfiles.kernel}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="weight" 
                    stroke="#ff7300" 
                    name="Kernel Weights"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    data={weightProfiles.attention}
                    dataKey="weight" 
                    stroke="#0088fe" 
                    name="Attention Weights"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="text-sm space-y-2">
            <p>
              Both kernel smoothing and attention compute weighted averages, but they differ in how they determine the weights:
            </p>
            <p>
              <span className="font-medium text-orange-600">Kernel Smoothing</span>: Weights 
              depend directly on distance between positions. Bandwidth controls 
              how quickly weights decay with distance.
            </p>
            <p>
              <span className="font-medium text-blue-600">Attention</span>: Weights come from 
              learned similarity in a transformed space. Temperature controls how 
              much the mechanism focuses on the highest similarities.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

document.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('attention-explorer-root');
  if (root) {
    ReactDOM.render(<AttentionKernelExplorer />, root);
  }
});

export default AttentionKernelExplorer;