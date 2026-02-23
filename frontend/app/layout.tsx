import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AstraQuant Dashboard",
  description: "Production-grade quantitative research dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-gray-900 antialiased">{children}</body>
    </html>
  );
}
