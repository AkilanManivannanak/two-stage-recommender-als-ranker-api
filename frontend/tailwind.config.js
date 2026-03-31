/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        cine: {
          bg: "#141414",
          surface: "#1a1a1a",
          card: "#222222",
          border: "#333333",
          accent: "#e50914",
          gold: "#f5f5f1",
          text: "#e5e5e5",
          muted: "#a3a3a3",
        },
      },
      boxShadow: {
        overlay: "0 20px 80px rgba(0,0,0,0.6)",
      },
      keyframes: {
        "slide-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "slide-up": "slide-up 0.3s ease-out",
      },
    },
  },
  plugins: [],
}
