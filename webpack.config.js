const path = require('path');

module.exports = {
  entry: './components/AttentionKernelExplorer/index.jsx',
  output: {
    filename: 'attention-explorer.bundle.js',
    path: path.resolve(__dirname, 'js'),
  },
  externals: {
    'react': 'window.React',
    'react-dom': 'window.ReactDOM'
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-react']
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader', 'postcss-loader']
      }
    ]
  },
  resolve: {
    extensions: ['.js', '.jsx']
  }
};