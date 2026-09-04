/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#05070a",
          900: "#0a0e14",
          800: "#10151d",
          700: "#171d27",
          600: "#232b38",
          500: "#3a4657",
        },
        ink: {
          400: "#6b7688",
          300: "#8b95a5",
          200: "#b3bcc9",
          100: "#dde2e9",
        },
        accent: {
          DEFAULT: "#3ecfb2",
          dim: "#1f6b5c",
        },
        warn: "#e0a940",
        crit: "#e05a4e",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        sans: ["'Inter'", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
