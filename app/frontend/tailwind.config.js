/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./index.html', './preview.html', './src/**/*.{js,ts,jsx,tsx}', './preview/**/*.{js,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--c-bg) / <alpha-value>)',
        surface: 'rgb(var(--c-surface) / <alpha-value>)',
        raised: 'rgb(var(--c-raised) / <alpha-value>)',
        sunken: 'rgb(var(--c-sunken) / <alpha-value>)',
        ink: 'rgb(var(--c-ink) / <alpha-value>)',
        muted: 'rgb(var(--c-muted) / <alpha-value>)',
        faint: 'rgb(var(--c-faint) / <alpha-value>)',
        line: 'rgb(var(--c-line) / <alpha-value>)',
        'line-strong': 'rgb(var(--c-line-strong) / <alpha-value>)',
        accent: 'rgb(var(--c-accent) / <alpha-value>)',
        'accent-ink': 'rgb(var(--c-accent-ink) / <alpha-value>)',
        'accent-soft': 'rgb(var(--c-accent-soft) / <alpha-value>)',

        // Semantic status. Replaces every hardcoded emerald/rose/amber.
        ok: 'rgb(var(--c-ok) / <alpha-value>)',
        'ok-soft': 'rgb(var(--c-ok-soft) / <alpha-value>)',
        warn: 'rgb(var(--c-warn) / <alpha-value>)',
        'warn-soft': 'rgb(var(--c-warn-soft) / <alpha-value>)',
        danger: 'rgb(var(--c-danger) / <alpha-value>)',
        'danger-soft': 'rgb(var(--c-danger-soft) / <alpha-value>)',

        // Themed log console, replacing hardcoded slate-*/bg-[#0d1118].
        console: 'rgb(var(--c-console) / <alpha-value>)',
        'console-ink': 'rgb(var(--c-console-ink) / <alpha-value>)',
        'console-muted': 'rgb(var(--c-console-muted) / <alpha-value>)',
        'console-line': 'rgb(var(--c-console-line) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['Geist', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Geist', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"Geist Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        soft: '0 1px 2px rgb(9 9 11 / 0.04), 0 10px 30px rgb(9 9 11 / 0.06)',
        lift: '0 2px 6px rgb(9 9 11 / 0.08), 0 24px 60px rgb(9 9 11 / 0.16)',
      },
    },
  },
  plugins: [],
}
