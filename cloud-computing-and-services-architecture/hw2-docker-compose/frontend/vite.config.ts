import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    // Local dev only: forward /api to the backend so the browser stays same-origin.
    // In Docker the proxy container does this job instead.
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
