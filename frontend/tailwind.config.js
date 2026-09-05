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
          // 400 was #6b7688 -- 3.99:1 against base-800 and 4.21:1 against
          // base-900, both under WCAG AA's 4.5:1 floor for normal text.
          // Every StatCard/panel label in the app uses text-ink-400 at
          // 10-11px, so this affected almost every label on the page.
          // #7a869a clears AA on both surfaces (4.97:1 / 5.25:1) while
          // staying visually distinct from ink-300.
          400: "#7a869a",
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
