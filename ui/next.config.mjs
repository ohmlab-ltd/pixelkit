/** @type {import('next').NextConfig} */
const nextConfig = {
  // Portable build: the UI compiles to a static out/ dir that the engine
  // serves itself (same-origin). `headers()` is unsupported in export mode
  // and the image optimizer needs a server, so images are unoptimized.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
