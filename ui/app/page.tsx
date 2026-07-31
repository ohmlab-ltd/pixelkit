"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

// Portable build has no marketing landing page — the root goes straight
// to the workspace. Client-side replace because `output: "export"` can't
// do server redirects.
export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/app");
  }, [router]);
  return null;
}
