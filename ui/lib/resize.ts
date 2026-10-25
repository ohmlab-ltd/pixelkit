// Image-file detection for uploads. The extension list applies only when the MIME type is missing.
const IMAGE_EXT_RE = /\.(jpe?g|png|webp|gif|bmp|tiff?|heic|heif|avif)$/i;
export function isImageFile(file: File): boolean {
  if (file.type && file.type.startsWith("image/")) return true;
  if (!file.type) return IMAGE_EXT_RE.test(file.name);
  return false;
}
