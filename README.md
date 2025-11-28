# BinaryStateFNN

A beautiful, interactive neural network visualizer and trainer built with React and Canvas.

![BinaryStateFNN Preview](https://replit.com/public/images/opengraph.png)

## Features

- **Real-time Visualization**: Watch your neural network learn in real-time with animated connections and node activations.
- **Interactive Training**: Control the training process with Play/Pause, Reset, and adjust parameters on the fly.
- **Dynamic Target Functions**: Train your network to approximate various mathematical functions:
  - Wave (Sine/Cosine mix)
  - Step Function
  - Gaussian Peaks
  - Spiral
  - Chaos (High-frequency trig mix)
- **Adjustable Parameters**:
  - Learning Rate (0.001 - 0.5)
  - Hidden Nodes (2 - 16)
- **Visual Feedback**:
  - Real-time Loss Graph
  - Epoch Counter
  - Connection weights visualized by thickness and color
  - Prediction vs Target overlay

## Tech Stack

- **Frontend**: React, TypeScript, Vite
- **Styling**: Tailwind CSS
- **Graphics**: HTML5 Canvas API
- **Icons**: Lucide React

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Usage

1. Select a **Target Function** from the bottom panel.
2. Adjust **Hidden Nodes** and **Learning Rate** if desired.
3. Click **Train** to start the backpropagation process.
4. Watch the network attempt to fit the curve!
5. If the network gets stuck or explodes (NaN), click **Reset** and try a lower learning rate.

## License

MIT
