import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: { port: 5175, proxy: {
    '/api': 'http://127.0.0.1:8020', '/static': 'http://127.0.0.1:8020' } },
  build: { outDir: 'dist' },
})
