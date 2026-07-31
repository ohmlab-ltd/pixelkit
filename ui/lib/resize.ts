// Image-file detection helper. The client-side downscale/quality ladder
// that used to live here is gone — PixelKit uploads ORIGINAL file bytes
// at full resolution and the engine stores them as-is.

// Sniff whether a File is an image we want to route through the
// upload pipeline. Prefers `file.type`; falls back to a known
// extension whitelist when the MIME is missing (Safari edge cases).
const IMAGE_EXT_RE = /\.(jpe?g|png|webp|gif|bmp|tiff?|heic|heif|avif)$/i;
export function isImageFile(file: File): boolean {
  if (file.type && file.type.startsWith("image/")) return true;
  if (!file.type) return IMAGE_EXT_RE.test(file.name);
  return false;
}
