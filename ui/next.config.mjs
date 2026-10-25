/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static out/, served by the engine. The image optimizer needs a server.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
