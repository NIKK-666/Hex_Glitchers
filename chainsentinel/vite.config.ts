import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'
import { defineConfig } from 'vite'
import path from 'path' // 1. Import the Node path module

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  resolve: { // 2. Add path aliasing here
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
