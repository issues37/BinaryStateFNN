import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Zap, Brain, TrendingDown } from 'lucide-react';

const NeuralNetworkTrainer = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [hiddenNodes, setHiddenNodes] = useState(8);
  const [targetFunction, setTargetFunction] = useState('wave');
  const [showNetwork, setShowNetwork] = useState(true);
  
  const networkRef = useRef<any>(null);
  const targetPointsRef = useRef<any[]>([]);
  const lossHistoryRef = useRef<number[]>([]);
  const animationRef = useRef<number>(null);

  // Initialize neural network
  const initNetwork = () => {
    const w1 = Array(hiddenNodes).fill(0).map(() => Math.random() * 2 - 1);
    const b1 = Array(hiddenNodes).fill(0).map(() => Math.random() * 2 - 1);
    const w2 = Array(hiddenNodes).fill(0).map(() => Math.random() * 2 - 1);
    const b2 = Math.random() * 2 - 1;
    
    networkRef.current = { w1, b1, w2, b2 };
    setEpoch(0);
    lossHistoryRef.current = [];
  };

  // Generate target function
  const generateTarget = () => {
    const points = [];
    for (let i = 0; i <= 100; i++) {
      const x = i / 100;
      let y;
      
      switch(targetFunction) {
        case 'wave':
          y = 0.5 + 0.3 * Math.sin(3 * x * Math.PI * 2) + 0.15 * Math.cos(7 * x * Math.PI * 2);
          break;
        case 'step':
          y = x < 0.3 ? 0.2 : x < 0.7 ? 0.8 : 0.3;
          break;
        case 'peaks':
          y = Math.exp(-((x - 0.3) ** 2) / 0.02) * 0.8 + Math.exp(-((x - 0.7) ** 2) / 0.02) * 0.6;
          break;
        case 'spiral':
          y = 0.5 + 0.4 * Math.sin(10 * x * Math.PI) * (1 - x);
          break;
        case 'chaos':
          y = 0.5 + 0.3 * Math.sin(5 * x * Math.PI) * Math.cos(13 * x * Math.PI);
          break;
        default:
          y = 0.5 + 0.3 * Math.sin(3 * x * Math.PI * 2);
      }
      
      points.push({ x, y: Math.max(0, Math.min(1, y)) });
    }
    targetPointsRef.current = points;
  };

  // Activation functions
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const relu = (x: number) => Math.max(0, x);

  // Forward pass
  const predict = (x: number) => {
    const { w1, b1, w2, b2 } = networkRef.current;
    const hidden = w1.map((w: number, i: number) => sigmoid(w * x + b1[i]));
    let output = b2;
    for (let i = 0; i < hidden.length; i++) {
      output += w2[i] * hidden[i];
    }
    return Math.max(0, Math.min(1, output));
  };

  // Training step with proper backpropagation
  const trainStep = () => {
    const { w1, b1, w2, b2 } = networkRef.current;
    const lr = learningRate;
    let totalLoss = 0;

    targetPointsRef.current.forEach(({ x, y }) => {
      // Forward pass
      const hidden = w1.map((w: number, i: number) => sigmoid(w * x + b1[i]));
      let output = b2;
      for (let i = 0; i < hidden.length; i++) {
        output += w2[i] * hidden[i];
      }
      
      // Loss
      const error = y - output;
      totalLoss += error * error;
      
      // Backpropagation
      for (let i = 0; i < hiddenNodes; i++) {
        const h = hidden[i];
        w2[i] += lr * error * h;
        
        const delta = error * w2[i] * h * (1 - h);
        w1[i] += lr * delta * x;
        b1[i] += lr * delta;
      }
      networkRef.current.b2 += lr * error;
    });

    const avgLoss = totalLoss / targetPointsRef.current.length;
    setLoss(avgLoss);
    lossHistoryRef.current.push(avgLoss);
    if (lossHistoryRef.current.length > 100) lossHistoryRef.current.shift();
  };

  // Drawing function
  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    
    // Clear with gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, '#0f172a');
    gradient.addColorStop(1, '#1e293b');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width;
      const y = (i / 10) * height;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Draw target function
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 3;
    ctx.beginPath();
    targetPointsRef.current.forEach(({ x, y }: any, i: number) => {
      const px = x * width;
      const py = height - y * height;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    
    // Draw target points
    ctx.fillStyle = '#22d3ee';
    targetPointsRef.current.forEach(({ x, y }) => {
      const px = x * width;
      const py = height - y * height;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Draw prediction
    if (networkRef.current) {
      ctx.strokeStyle = '#f472b6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const y = predict(x);
        const px = x * width;
        const py = height - y * height;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    
    // Draw neural network diagram
    if (showNetwork && networkRef.current) {
      const netX = width - 180;
      const netY = 20;
      const netWidth = 160;
      const netHeight = 200;
      
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fillRect(netX, netY, netWidth, netHeight);
      ctx.strokeStyle = '#475569';
      ctx.strokeRect(netX, netY, netWidth, netHeight);
      
      // Draw nodes
      const inputX = netX + 30;
      const hiddenX = netX + 80;
      const outputX = netX + 130;
      
      const inputY = netY + netHeight / 2;
      
      // Input node
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(inputX, inputY, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Hidden nodes
      const { w1, w2 } = networkRef.current;
      for (let i = 0; i < Math.min(hiddenNodes, 6); i++) {
        const hy = netY + 30 + (i * (netHeight - 60)) / Math.max(1, Math.min(hiddenNodes, 6) - 1);
        
        // Connections from input
        const weight = w1[i];
        ctx.strokeStyle = weight > 0 ? `rgba(34, 211, 238, ${Math.abs(weight)})` : `rgba(244, 114, 182, ${Math.abs(weight)})`;
        ctx.lineWidth = Math.abs(weight) * 2;
        ctx.beginPath();
        ctx.moveTo(inputX, inputY);
        ctx.lineTo(hiddenX, hy);
        ctx.stroke();
        
        // Hidden node
        ctx.fillStyle = '#8b5cf6';
        ctx.beginPath();
        ctx.arc(hiddenX, hy, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Connections to output
        const weight2 = w2[i];
        ctx.strokeStyle = weight2 > 0 ? `rgba(34, 211, 238, ${Math.abs(weight2)})` : `rgba(244, 114, 182, ${Math.abs(weight2)})`;
        ctx.lineWidth = Math.abs(weight2) * 2;
        ctx.beginPath();
        ctx.moveTo(hiddenX, hy);
        ctx.lineTo(outputX, inputY);
        ctx.stroke();
      }
      
      // Output node
      ctx.fillStyle = '#f472b6';
      ctx.beginPath();
      ctx.arc(outputX, inputY, 8, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw loss graph
    if (lossHistoryRef.current.length > 1) {
      const graphX = 20;
      const graphY = height - 120;
      const graphWidth = 200;
      const graphHeight = 100;
      
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fillRect(graphX, graphY, graphWidth, graphHeight);
      ctx.strokeStyle = '#475569';
      ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);
      
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.beginPath();
      lossHistoryRef.current.forEach((l, i) => {
        const x = graphX + (i / lossHistoryRef.current.length) * graphWidth;
        const y = graphY + graphHeight - (1 - Math.min(l, 1)) * graphHeight;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      
      ctx.fillStyle = '#fbbf24';
      ctx.font = '12px monospace';
      ctx.fillText('Loss', graphX + 5, graphY + 15);
    }
  };

  // Training loop
  useEffect(() => {
    if (isTraining) {
      const train = () => {
        trainStep();
        setEpoch(e => e + 1);
        draw();
        animationRef.current = requestAnimationFrame(train);
      };
      animationRef.current = requestAnimationFrame(train);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      draw();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isTraining, showNetwork]);

  // Initialize on mount and when parameters change
  useEffect(() => {
    generateTarget();
    initNetwork();
    draw();
  }, [targetFunction, hiddenNodes]);

  const reset = () => {
    setIsTraining(false);
    generateTarget();
    initNetwork();
    draw();
  };

  return (
    <div className="w-full min-h-screen bg-slate-900 text-white p-6 flex flex-col font-sans">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Brain className="w-8 h-8 text-cyan-400" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-pink-400 bg-clip-text text-transparent">
            BinaryStateFNN
          </h1>
        </div>
        <div className="flex items-center gap-4 bg-slate-800 px-4 py-2 rounded-lg border border-slate-700">
          <TrendingDown className="w-5 h-5 text-yellow-400" />
          <div>
            <div className="text-xs text-slate-400">Loss</div>
            <div className="text-lg font-bold text-yellow-400">{loss.toFixed(4)}</div>
          </div>
          <div className="ml-4">
            <div className="text-xs text-slate-400">Epoch</div>
            <div className="text-lg font-bold text-cyan-400">{epoch}</div>
          </div>
        </div>
      </div>

      <div className="flex-1 bg-slate-800 rounded-lg overflow-hidden shadow-2xl border border-slate-700 relative">
        <canvas
          ref={canvasRef}
          width={1200}
          height={600}
          className="w-full h-full object-contain"
        />
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 shadow-lg">
          <h3 className="text-sm font-semibold mb-3 text-cyan-400 uppercase tracking-wider">Training Controls</h3>
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setIsTraining(!isTraining)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded font-semibold transition-all shadow-lg ${
                isTraining
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white'
              }`}
            >
              {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isTraining ? 'Pause' : 'Train'}
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded font-semibold transition-all flex items-center gap-2 text-white shadow-lg"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="text-xs text-slate-400 block mb-2 flex justify-between">
                <span>Learning Rate</span>
                <span className="font-mono text-cyan-400">{learningRate.toFixed(3)}</span>
              </label>
              <input
                type="range"
                min="0.001"
                max="0.5"
                step="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
            
            <div>
              <label className="text-xs text-slate-400 block mb-2 flex justify-between">
                <span>Hidden Nodes</span>
                <span className="font-mono text-cyan-400">{hiddenNodes}</span>
              </label>
              <input
                type="range"
                min="2"
                max="16"
                step="1"
                value={hiddenNodes}
                onChange={(e) => setHiddenNodes(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
          </div>
        </div>

        <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 shadow-lg">
          <h3 className="text-sm font-semibold mb-3 text-pink-400 uppercase tracking-wider">Target Function</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4">
            {['wave', 'step', 'peaks', 'spiral', 'chaos'].map((func) => (
              <button
                key={func}
                onClick={() => setTargetFunction(func)}
                className={`px-3 py-2 rounded text-sm font-medium transition-all shadow-md ${
                  targetFunction === func
                    ? 'bg-gradient-to-r from-pink-500 to-purple-500 text-white ring-1 ring-purple-300'
                    : 'bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-white'
                }`}
              >
                {func.charAt(0).toUpperCase() + func.slice(1)}
              </button>
            ))}
          </div>
          
          <div className="flex items-center gap-3 mt-4 pt-4 border-t border-slate-700">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="showNetwork"
                checked={showNetwork}
                onChange={(e) => setShowNetwork(e.target.checked)}
                className="w-4 h-4 rounded border-slate-600 text-purple-500 focus:ring-purple-500 bg-slate-700"
              />
              <label htmlFor="showNetwork" className="text-sm text-slate-300 flex items-center gap-2 cursor-pointer select-none">
                <Zap className="w-4 h-4 text-purple-400" />
                Show Network Diagram
              </label>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 text-center text-xs text-slate-500 font-mono">
        <span className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-400"></span> Target Function</span>
        <span className="mx-2">•</span>
        <span className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-pink-400"></span> Neural Network Prediction</span>
      </div>
    </div>
  );
};

export default NeuralNetworkTrainer;
