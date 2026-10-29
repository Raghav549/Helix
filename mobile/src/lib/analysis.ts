/** Turn the /api/image/analyze payload into readable text, without inventing values. */

export function formatAnalysis(payload: Record<string, unknown> | null, fileName = 'image'): string {
  if (!payload || typeof payload !== 'object') return `No analysis was returned for ${fileName}.`;
  if (typeof payload.error === 'string' && payload.error) return `Analysis failed: ${payload.error}`;

  const lines: string[] = [`**${fileName}**`];
  const push = (label: string, value: unknown) => {
    if (value === null || value === undefined || value === '') return;
    if (typeof value === 'object') {
      lines.push(`• ${label}: ${JSON.stringify(value)}`);
      return;
    }
    lines.push(`• ${label}: ${String(value)}`);
  };

  push('Description', payload.description);
  push('Average colour', payload.average_color ?? payload.mean_color);
  push('Brightness', payload.brightness);
  push('Contrast', payload.contrast);
  push('Dominant shapes', payload.shapes);
  push('Dominant colours', payload.colors ?? payload.dominant_colors);
  push('Vision model output', payload.model_output ?? payload.prediction ?? payload.label);
  push('Caption (synthetic-domain model)', payload.synthetic_domain_caption);

  if (typeof payload.caption_warning === 'string') lines.push(`\n${payload.caption_warning}`);
  if (Array.isArray(payload.warnings)) {
    for (const warning of payload.warnings) lines.push(`⚠ ${String(warning)}`);
  }
  if (lines.length === 1) {
    lines.push('The analysis tool returned no recognised fields, so nothing is shown rather than a guess.');
  }
  lines.push('\nAnalysed by the local image tool, not by the chat model.');
  return lines.join('\n');
}

/** Messages the app can send for a quick action, kept honest about what runs. */
export function quickActionPrompt(id: 'image' | 'summarize' | 'code' | 'brainstorm' | 'analyze' | 'more'): string {
  switch (id) {
    case 'summarize':
      return 'Summarize the following text in five bullet points:\n\n';
    case 'code':
      return 'Write a Python function that ';
    case 'brainstorm':
      return 'Give me five distinct ideas for ';
    case 'analyze':
      return 'Analyze the attached image and describe what you see.';
    case 'more':
      return '';
    case 'image':
    default:
      return '';
  }
}
