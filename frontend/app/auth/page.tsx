"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { signUp, logIn } from "@/lib/api";

type Mode = "signin" | "login";

export default function AuthPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("signin");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setLoading(true);
    try {
      if (mode === "signin") {
        const res = await signUp(username, password, email || undefined);
        if (typeof window !== "undefined" && res?.username) window.sessionStorage.setItem("userUsername", res.username);
        setMessage({ type: "success", text: "Account created! Set up your profile..." });
        setPassword("");
        setTimeout(() => router.push("/onboarding"), 800);
      } else {
        await logIn(username, password);
        if (typeof window !== "undefined" && username) window.sessionStorage.setItem("userUsername", username);
        router.push("/");
      }
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "Something went wrong" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="w-full max-w-sm">
        <Link href="/" className="inline-flex items-center gap-2 text-forest-700 hover:text-forest-800 mb-8">
          <span className="text-xl">🌿</span>
          <span className="font-medium">How to Keep Your Plants Alive</span>
        </Link>

        <div className="bg-white/80 backdrop-blur rounded-2xl shadow-leaf border border-sage-200/60 p-8">
          <h1 className="text-xl font-semibold text-forest-800 mb-6">
            {mode === "signin" ? "Create account" : "Welcome back"}
          </h1>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-forest-700 mb-1">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="w-full px-4 py-2.5 rounded-lg border border-sage-300 bg-white text-forest-800 placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent"
                placeholder="your username"
              />
            </div>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-forest-700 mb-1">
                Email (optional)
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border border-sage-300 bg-white text-forest-800 placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-forest-700 mb-1">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-2.5 rounded-lg border border-sage-300 bg-white text-forest-800 placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent"
                placeholder="••••••••"
              />
            </div>

            {message && (
              <p
                className={`text-sm ${
                  message.type === "success" ? "text-forest-600" : "text-rose-600"
                }`}
              >
                {message.text}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-lg bg-forest-600 text-white font-medium hover:bg-forest-700 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:ring-offset-2 disabled:opacity-60 transition-colors"
            >
              {loading ? "..." : mode === "signin" ? "Sign up" : "Log in"}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-forest-600">
            {mode === "signin" ? "Already have an account?" : "Need an account?"}{" "}
            <button
              type="button"
              onClick={() => {
                setMode(mode === "signin" ? "login" : "signin");
                setMessage(null);
              }}
              className="font-medium text-forest-700 hover:text-forest-800 underline"
            >
              {mode === "signin" ? "Log in" : "Sign up"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
