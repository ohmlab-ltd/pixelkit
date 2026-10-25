/** Label names from typed or pasted text: split on commas and newlines, trailing full stops dropped. */
export function splitLabels(raw: string): string[] {
  return raw
    .split(/[,\n]/)
    .map((part) => part.trim().replace(/\.+$/, "").trim())
    .filter(Boolean);
}
