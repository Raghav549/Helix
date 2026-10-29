import { useColorScheme } from 'react-native';
import { buildTheme, type Theme, type ThemeMode } from '../theme/tokens';

export function useTheme(mode: ThemeMode = 'system'): Theme {
  const scheme = useColorScheme();
  const dark = mode === 'dark' || (mode === 'system' && scheme === 'dark');
  return buildTheme(dark);
}
