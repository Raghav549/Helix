import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { buildTheme, palette } from '../src/theme/tokens';

const css = readFileSync(join(__dirname, '..', '..', '..', 'public', 'static', 'styles.css'), 'utf8');

function cssVar(name: string): string {
  const match = css.match(new RegExp(`--${name}:\\s*([^;]+);`));
  return (match?.[1] ?? '').trim().toLowerCase();
}

test('mobile palette matches the web design tokens', () => {
  const pairs: Array<[string, string]> = [
    ['paper', palette.paper],
    ['paper-soft', palette.paperSoft],
    ['paper-tint', palette.paperTint],
    ['ink', palette.ink],
    ['ink-soft', palette.inkSoft],
    ['muted', palette.muted],
    ['muted-soft', palette.mutedSoft],
    ['line', palette.line],
    ['line-strong', palette.lineStrong],
    ['pink', palette.pink],
    ['magenta', palette.magenta],
    ['coral', palette.coral],
    ['orange', palette.orange],
    ['success', palette.success],
    ['warning', palette.warning],
    ['danger', palette.danger],
  ];
  for (const [name, value] of pairs) {
    assert.equal(cssVar(name), value.toLowerCase(), `--${name} drifted between web and mobile`);
  }
});

test('theme switching keeps accents and swaps surfaces', () => {
  const light = buildTheme(false);
  const dark = buildTheme(true);
  assert.equal(light.mode, 'light');
  assert.equal(dark.mode, 'dark');
  assert.equal(dark.palette.pink, light.palette.pink);
  assert.notEqual(dark.palette.paper, light.palette.paper);
  assert.equal(dark.radii.lg, light.radii.lg);
});
