/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: { brand: { 50: '#eef2ff', 100: '#e0e7ff', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' } },
      boxShadow: { card: '0 1px 2px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.06)' },
    },
  },
  plugins: [],
}
