/**
 * Design tokens for the Android client.
 *
 * These mirror public/static/styles.css. A regression test
 * (tests/tokens.test.ts) parses the CSS custom properties and fails when a
 * shared value drifts, so the two clients cannot silently diverge.
 */
export interface Palette {
  pink: string;
  magenta: string;
  coral: string;
  orange: string;
  ink: string;
  inkSoft: string;
  muted: string;
  mutedSoft: string;
  paper: string;
  paperSoft: string;
  paperTint: string;
  line: string;
  lineStrong: string;
  success: string;
  warning: string;
  danger: string;
}

export const palette: Palette = {
  pink: '#ff2f86',
  magenta: '#e23aa9',
  coral: '#ff6a56',
  orange: '#ffa63d',
  ink: '#0c0c0f',
  inkSoft: '#22222a',
  muted: '#6d6d78',
  mutedSoft: '#93939e',
  paper: '#ffffff',
  paperSoft: '#fbf9f9',
  paperTint: '#f7f4f6',
  line: '#ece9ec',
  lineStrong: '#ded9de',
  success: '#17a673',
  warning: '#c9761a',
  danger: '#d8385a',
} as const;

export const darkPalette = {
  paper: '#0b0b10',
  paperSoft: '#111117',
  paperTint: '#16161d',
  ink: '#f6f4f6',
  inkSoft: '#e7e4ea',
  muted: '#a4a1ad',
  mutedSoft: '#7c7986',
  line: '#23232b',
  lineStrong: '#2f2f39',
} as const;

export const radii = { sm: 10, md: 14, lg: 20, xl: 28, pill: 999 } as const;

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32, xxxl: 48 } as const;

export const fontSizes = { xs: 11, sm: 13, base: 15, md: 17, lg: 22, xl: 30 } as const;

export const motion = { fast: 140, base: 240, slow: 480 } as const;

export const gradients = {
  wave: ['rgba(255,47,134,0.55)', 'rgba(255,106,86,0.5)', 'rgba(255,166,61,0.45)'] as const,
  primary: ['#ff2f86', '#ff6a56', '#ffa63d'] as const,
  soft: ['rgba(255,47,134,0.14)', 'rgba(255,166,61,0.16)'] as const,
};

export type ThemeMode = 'light' | 'dark' | 'system';

export interface Theme {
  mode: 'light' | 'dark';
  palette: Palette;
  radii: typeof radii;
  spacing: typeof spacing;
  fontSizes: typeof fontSizes;
  gradients: typeof gradients;
  motion: typeof motion;
}

export function buildTheme(dark: boolean): Theme {
  return {
    mode: dark ? 'dark' : 'light',
    palette: dark ? { ...palette, ...darkPalette } : { ...palette },
    radii,
    spacing,
    fontSizes,
    gradients,
    motion,
  };
}
