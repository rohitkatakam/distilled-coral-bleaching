/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        coral: {
          50: '#fff5f2',
          100: '#ffe8e1',
          200: '#ffd4c6',
          300: '#ffb59d',
          400: '#ff8a6a',
          500: '#ff633e', // Base coral
          600: '#ee431f',
          700: '#c53012',
          800: '#a32a13',
          900: '#842616',
        },
        ocean: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e', // Deep ocean
          950: '#082f49',
        },
      },
    },
  },
  plugins: [],
}
