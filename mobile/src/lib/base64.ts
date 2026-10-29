/** Minimal base64 encoder for HTTP Basic auth headers (no Buffer in RN). */
const TABLE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

export function encodeBase64(input: string): string {
  const bytes = toUtf8(input);
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i] ?? 0;
    const b1 = bytes[i + 1];
    const b2 = bytes[i + 2];
    const triplet = (b0 << 16) | ((b1 ?? 0) << 8) | (b2 ?? 0);
    out += TABLE[(triplet >> 18) & 63];
    out += TABLE[(triplet >> 12) & 63];
    out += b1 === undefined ? '=' : TABLE[(triplet >> 6) & 63];
    out += b2 === undefined ? '=' : TABLE[triplet & 63];
  }
  return out;
}

export function basicAuthHeader(username: string, password: string): string {
  return `Basic ${encodeBase64(`${username}:${password}`)}`;
}

function toUtf8(value: string): number[] {
  const out: number[] = [];
  for (const char of value) {
    const code = char.codePointAt(0) ?? 0;
    if (code < 0x80) out.push(code);
    else if (code < 0x800) out.push(0xc0 | (code >> 6), 0x80 | (code & 63));
    else if (code < 0x10000) out.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
    else
      out.push(
        0xf0 | (code >> 18),
        0x80 | ((code >> 12) & 63),
        0x80 | ((code >> 6) & 63),
        0x80 | (code & 63),
      );
  }
  return out;
}
