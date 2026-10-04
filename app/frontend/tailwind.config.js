/** @type {import('tailwindcss').Config} */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`

// Design tokens come from design/DESIGN.md; the actual colours live in CSS variables (src/index.css) so that
// light and dark mode switch by toggling one class on <html>.
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        canvas: v('canvas'), surface: v('surface'), subdued: v('subdued'), stage: v('stage'), line: v('line'),
        ink: { DEFAULT: v('ink'), 2: v('ink-2'), 3: v('ink-3') },
        brand: { DEFAULT: v('primary'), hover: v('primary-hover'), tint: v('tint'), 'on-tint': v('on-tint') },
        good: { DEFAULT: v('emerald'), tint: v('emerald-tint') },
        info: { DEFAULT: v('sky'), tint: v('sky-tint') },
        warn: { DEFAULT: v('amber'), tint: v('amber-tint') },
        bad: { DEFAULT: v('red'), tint: v('red-tint') },
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans Variable"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['"Inter Variable"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono Variable"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      borderRadius: { sm: '0.375rem', DEFAULT: '0.5rem', md: '0.75rem', lg: '1rem', xl: '1.5rem', '2xl': '1.5rem' },
      boxShadow: {
        card: '0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.06)',
        float: '0 10px 24px rgba(15,23,42,.12)',
        glow: '0 4px 14px rgba(79,70,229,.28)',
      },
      keyframes: {
        fadeUp: { '0%': { opacity: 0, transform: 'translateY(8px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        shimmer: { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
        pulseDot: { '0%,100%': { opacity: 1 }, '50%': { opacity: 0.35 } },
      },
      animation: {
        'fade-up': 'fadeUp 300ms cubic-bezier(0.2,0,0,1) both',
        shimmer: 'shimmer 1.6s linear infinite',
        'pulse-dot': 'pulseDot 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
