"use client";

import Link from "next/link";
import { getToken } from "@/lib/api";

export default function AgentPage() {
  const isLoggedIn = !!getToken();

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">Please sign in to use the agent.</p>
        <Link href="/auth" className="text-forest-700 font-medium underline">
          Sign in
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-2xl mx-auto text-center">
        <h1 className="text-2xl font-semibold text-forest-800 mb-2">Agent</h1>
        <p className="text-forest-600">Coming soon.</p>
      </div>
    </div>
  );
}
