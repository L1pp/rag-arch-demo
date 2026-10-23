import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const apiProxy = {
  '/api': {
    target: process.env.RAG_ARCH_DEV_API_TARGET || 'http://127.0.0.1:8000',
    changeOrigin: false,
  },
}

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // The browser owns every page route, including /login. Only API calls
      // cross to the separately running backend during development.
      ...apiProxy,
    },
  },
  preview: {
    proxy: {
      ...apiProxy,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
