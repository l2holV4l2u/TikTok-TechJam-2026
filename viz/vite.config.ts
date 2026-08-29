import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { runsPlugin } from "./vite-plugin-runs";

export default defineConfig({
  plugins: [react(), runsPlugin()],
});
