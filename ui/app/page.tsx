"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

// Client-side redirect to /app: `output: "export"` has no server redirects.
export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/app");
  }, [router]);
  return null;
}
