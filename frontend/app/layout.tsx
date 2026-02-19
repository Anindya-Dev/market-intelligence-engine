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
    <html lang="en" className="dark">
      <body className="min-h-screen bg-bg text-gray-100 antialiased">{children}</body>
    </html>
  );
}
