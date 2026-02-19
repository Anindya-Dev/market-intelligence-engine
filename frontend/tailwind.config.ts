import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0f172a",
        panel: "#111827",
        border: "#1f2937",
        muted: "#9ca3af",
        bull: "#22c55e",
        bear: "#ef4444",
      },
    },
  },
  plugins: [],
};

export default config;
