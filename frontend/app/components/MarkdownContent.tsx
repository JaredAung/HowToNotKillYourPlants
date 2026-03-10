"use client";

/**
 * Lightweight markdown renderer for chat messages.
 * Renders **bold**, *italic*, bullet lists, and line breaks.
 */
export function MarkdownContent({ content }: { content: string }) {
  if (!content) return null;

  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];
  let inList = false;

  const isPlantName = (item: string) => {
    const t = item.trim();
    return t.endsWith(":") && t.includes("(") && t.includes(")");
  };

  const flushList = () => {
    if (listItems.length === 0) {
      inList = false;
      return;
    }
    const out: React.ReactNode[] = [];
    let bulletBatch: string[] = [];

    const flushBullets = () => {
      if (bulletBatch.length > 0) {
        out.push(
          <ul key={out.length} className="list-disc list-inside pl-2 space-y-1 my-1">
            {bulletBatch.map((item, j) => (
              <li key={j} className="text-inherit">
                <InlineMarkdown text={item} />
              </li>
            ))}
          </ul>
        );
        bulletBatch = [];
      }
    };

    listItems.forEach((item) => {
      if (isPlantName(item)) {
        flushBullets();
        out.push(
          <div
            key={out.length}
            className="text-lg font-bold mt-4 mb-1 first:mt-0 text-forest-900"
          >
            <InlineMarkdown text={item.replace(/:$/, "")} />
          </div>
        );
      } else {
        bulletBatch.push(item);
      }
    });
    flushBullets();
    elements.push(
      <div key={elements.length} className="my-2">
        {out}
      </div>
    );
    listItems = [];
    inList = false;
  };

  const parseLine = (line: string) => {
    const trimmed = line.trim();
    const listMatch = trimmed.match(/^[*\-]\s+(.*)$/);
    if (listMatch) {
      if (!inList) flushList();
      inList = true;
      listItems.push(listMatch[1]);
      return;
    }
    flushList();
    if (trimmed) {
      elements.push(
        <p key={elements.length} className="my-1.5 first:mt-0 last:mb-0">
          <InlineMarkdown text={trimmed} />
        </p>
      );
    } else if (elements.length > 0) {
      elements.push(<br key={elements.length} />);
    }
  };

  lines.forEach(parseLine);
  flushList();

  return <div className="markdown-content">{elements}</div>;
}

/** Renders **bold** and *italic* inline. */
function InlineMarkdown({ text }: { text: string }) {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*([^*]+)\*\*/);
    const italicMatch = remaining.match(/\*([^*]+)\*/);

    let match: RegExpMatchArray | null = null;
    let type: "bold" | "italic" | "text" = "text";
    let matchIndex = Infinity;

    if (boldMatch && boldMatch.index !== undefined) {
      matchIndex = boldMatch.index;
      match = boldMatch;
      type = "bold";
    }
    if (italicMatch && italicMatch.index !== undefined && italicMatch.index < matchIndex) {
      matchIndex = italicMatch.index;
      match = italicMatch;
      type = "italic";
    }

    if (match && match.index !== undefined) {
      if (match.index > 0) {
        parts.push(
          <span key={key++}>{remaining.slice(0, match.index)}</span>
        );
      }
      if (type === "bold") {
        parts.push(<strong key={key++} className="font-semibold">{match[1]}</strong>);
      } else {
        parts.push(<em key={key++} className="italic">{match[1]}</em>);
      }
      remaining = remaining.slice(match.index + match[0].length);
    } else {
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }
  }

  return <>{parts}</>;
}
