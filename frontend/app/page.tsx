import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-bg">
      <Link
        href="/dashboard"
        className="rounded border border-border bg-panel px-4 py-2 text-sm text-gray-100"
      >
        Open Dashboard
      </Link>
    </main>
  );
}
