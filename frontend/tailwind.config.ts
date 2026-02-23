import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#f5f7fb",
        panel: "#ffffff",
        border: "#d9dee7",
        muted: "#6b7280",
        bull: "#22c55e",
        bear: "#ef4444",
      },
    },
  },
  plugins: [],
};

export default config;
