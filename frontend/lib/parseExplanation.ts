/** Generic closing sentences the LLM sometimes adds to each plant—strip from plant text. */
const GENERIC_CLOSING_PATTERNS = [
  /\s*Remember,?\s*every plant is unique[\s\S]*?[.!]\s*$/i,
  /\s*Make sure to research and understand the specific needs[\s\S]*?[.!]\s*$/i,
  /\s*Do your research before bringing[\s\S]*?[.!]\s*$/i,
  /\s*For this user,?\s*every plant has unique characteristics[\s\S]*?[.!]\s*$/i,
];

function stripGenericClosingFromPlant(text: string): string {
  let out = text.trim();
  for (const re of GENERIC_CLOSING_PATTERNS) {
    out = out.replace(re, "").trim();
  }
  return out;
}

/** Parse LLM explanation into intro, plant items, and closing. */
export function parseExplanation(text: string): {
  intro?: string;
  items: { name: string; latin?: string; text: string }[];
  closing?: string;
} {
  const items: { name: string; latin?: string; text: string }[] = [];
  let closing: string | undefined;
  const parts = text.split(/\n•\s*\*\*/);
  const intro = parts[0]?.trim() || undefined;

  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]!;
    const colonIdx = part.indexOf("**: ");
    if (colonIdx < 0) continue;
    const fullName = part.slice(0, colonIdx).trim();
    const rest = part.slice(colonIdx + 4);
    const paren = fullName.indexOf(" (");
    const name = paren > 0 ? fullName.slice(0, paren) : fullName;
    const latin = paren > 0 ? fullName.slice(paren + 2, -1) : undefined;

    const closingMatch = rest.match(/\n\n(All of these|Based on this|I would rank|These plants)([\s\S]*)/i);
    let explanation: string;
    if (closingMatch && closingMatch.index != null) {
      explanation = rest.slice(0, closingMatch.index).trim();
      closing = (closingMatch[1] + closingMatch[2]).trim();
    } else {
      const paras = rest.split(/\n\n+/);
      const last = paras[paras.length - 1];
      if (last && /^(All|Based|I would|These|In summary)/i.test(last)) {
        explanation = paras.slice(0, -1).join("\n\n").trim();
        closing = last;
      } else {
        explanation = rest.trim();
      }
    }
    explanation = stripGenericClosingFromPlant(explanation);
    if (!closing) {
      const genericMatch = rest.match(/(Remember,?\s*every plant is unique[\s\S]*?[.!])/i)
        || rest.match(/(For this user,?\s*every plant has unique characteristics[\s\S]*?[.!])/i);
      if (genericMatch) closing = genericMatch[1].trim();
    }
    items.push({ name, latin, text: explanation });
  }

  if (!closing) {
    const m = text.match(/\n\n(All of these[\s\S]+)/);
    if (m) closing = m[1].trim();
  }
  return { intro, items, closing };
}
