/** Very small code-fence splitter so replies render code in a separate surface. */
export function splitCodeBlocks(content: string): Array<{ kind: 'text' | 'code'; value: string }> {
  const parts: Array<{ kind: 'text' | 'code'; value: string }> = [];
  const pattern = /```[a-zA-Z0-9_+-]*\n?([\s\S]*?)```/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(content)) !== null) {
    if (match.index > last) parts.push({ kind: 'text', value: content.slice(last, match.index) });
    parts.push({ kind: 'code', value: (match[1] ?? '').replace(/\n$/, '') });
    last = match.index + match[0].length;
  }
  if (last < content.length) parts.push({ kind: 'text', value: content.slice(last) });
  return parts.filter((part) => part.value.trim().length > 0);
}
