import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { SessionProviderWrapper } from "./SessionProviderWrapper";
import { MobileBlock } from "./MobileBlock";
import { ThemeProvider } from "./ThemeProvider";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

const SITE_URL = (
  process.env.NEXT_PUBLIC_APP_URL
  ?? "https://pixelkit.ai"
).replace(/\/+$/, "");

const SITE_DESCRIPTION =
  "Pixel Kit auto-labels images, generates augmentations, and exports training-ready datasets for computer vision AI. Free tier, no credit card required.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Pixel Kit · Vision AI Labelling & Training Platform",
    template: "%s · Pixel Kit",
  },
  description: SITE_DESCRIPTION,
  applicationName: "Pixel Kit",
  authors: [{ name: "Ohm Lab Ltd" }],
  creator: "Ohm Lab Ltd",
  publisher: "Ohm Lab Ltd",
  keywords: [
    "computer vision",
    "auto-labelling",
    "image annotation",
    "object detection",
    "segmentation",
    "dataset augmentation",
    "training dataset",
    "machine learning",
  ],
  alternates: { canonical: SITE_URL },
  openGraph: {
    type: "website",
    url: SITE_URL,
    title: "Pixel Kit · Vision AI Labelling & Training Platform",
    description: SITE_DESCRIPTION,
    siteName: "Pixel Kit",
    images: [
      {
        url: "/opengraph-image",
        width: 1200,
        height: 630,
        alt: "Pixel Kit · Vision AI Labelling & Training Platform",
      },
    ],
    locale: "en_GB",
  },
  twitter: {
    card: "summary_large_image",
    title: "Pixel Kit · Vision AI Labelling & Training Platform",
    description: SITE_DESCRIPTION,
    images: ["/opengraph-image"],
  },
  icons: {
    icon: "/pixelkit-favicon.svg",
    apple: "/pixelkit-favicon.svg",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
};

// JSON-LD structured data. Inlined in the document <head> below so
// Google can parse Organization + SoftwareApplication signals on
// every route without needing a per-page repeat.
const ORGANIZATION_LD = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: "Pixel Kit",
  url: SITE_URL,
  logo: `${SITE_URL}/pixelkit-favicon.svg`,
  sameAs: [] as string[],
};

const SOFTWARE_LD = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "Pixel Kit",
  applicationCategory: "BusinessApplication",
  operatingSystem: "Web",
  description: SITE_DESCRIPTION,
  url: SITE_URL,
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "GBP",
    availability: "https://schema.org/InStock",
  },
};

// Inline pre-hydration script: applies the persisted theme class on
// <html> before React mounts so the first paint matches the user's
// stored preference. Without this, users on light mode see a brief
// dark flash on every full page load.
//
// First-visit default is LIGHT, t === null when no preference is
// stored yet, so the only way the page renders dark on first paint
// is if the user explicitly toggled into dark on a previous visit
// (in which case t === 'dark').
const themeInitScript = `
(function() {
  try {
    var t = window.localStorage.getItem('pixelkit-theme');
    var dark = t === 'dark';
    var root = document.documentElement;
    if (dark) root.classList.add('dark'); else root.classList.remove('dark');
    root.style.colorScheme = dark ? 'dark' : 'light';
  } catch (e) {
    /* localStorage blocked, fall back to light to match the default */
  }
})();
`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" style={{ colorScheme: "light" }} suppressHydrationWarning>
      <head>
        {/* eslint-disable-next-line @next/next/no-sync-scripts */}
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        {/* JSON-LD: Organization + SoftwareApplication. Google reads
            these to power knowledge-panel + rich-result eligibility. */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(ORGANIZATION_LD) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(SOFTWARE_LD) }}
        />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased font-sans`}>
        <ThemeProvider>
          <SessionProviderWrapper>{children}</SessionProviderWrapper>
          <MobileBlock />
        </ThemeProvider>
      </body>
    </html>
  );
}
