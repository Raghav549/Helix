#!/usr/bin/env node
/**
 * Offline replacement for `expo install --check`.
 *
 * Compares every installed Expo-managed dependency with the versions that the
 * installed Expo SDK expects (node_modules/expo/bundledNativeModules.json).
 * No network access needed, so it works in restricted CI and on a laptop.
 */
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const bundledPath = join(root, 'node_modules', 'expo', 'bundledNativeModules.json');

if (!existsSync(bundledPath)) {
  console.error('Install dependencies first (npm ci): expo/bundledNativeModules.json is missing.');
  process.exit(1);
}

const bundled = JSON.parse(readFileSync(bundledPath, 'utf8'));
const declared = { ...pkg.dependencies, ...pkg.devDependencies };

function versionOf(name) {
  const installed = join(root, 'node_modules', name, 'package.json');
  if (existsSync(installed)) return JSON.parse(readFileSync(installed, 'utf8')).version;
  return declared[name] ?? null;
}

function parse(version) {
  const match = /^[~^]?(\d+)\.(\d+)\.(\d+)/.exec(String(version ?? ''));
  return match ? { major: +match[1], minor: +match[2], patch: +match[3] } : null;
}

function satisfies(actual, expected) {
  const want = parse(expected);
  const got = parse(actual);
  if (!want || !got) return true; // unparseable ranges are not our business
  if (String(expected).startsWith('~')) {
    return got.major === want.major && got.minor === want.minor;
  }
  if (String(expected).startsWith('^')) {
    return want.major === 0 ? got.major === 0 && got.minor === want.minor : got.major === want.major;
  }
  return got.major === want.major;
}

const problems = [];
for (const [name, expected] of Object.entries(bundled)) {
  if (!declared[name]) continue;
  if (!satisfies(declared[name], expected)) {
    problems.push(`${name}: package.json asks for ${declared[name]}, the SDK expects ${expected}`);
  }
  const actual = versionOf(name);
  if (!actual) {
    problems.push(`${name}: declared but not installed`);
    continue;
  }
  if (!satisfies(actual, expected)) {
    problems.push(`${name}: installed ${actual} does not match the SDK's ${expected}`);
  }
}

if (problems.length) {
  console.error('Expo SDK dependency mismatch:');
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exit(1);
}

console.log(`All Expo-managed dependencies match SDK ${JSON.parse(readFileSync(join(root, 'node_modules', 'expo', 'package.json'), 'utf8')).version}.`);
