import { PropsWithChildren, ReactNode } from "react";

interface CardProps extends PropsWithChildren {
  title: string;
  rightSlot?: ReactNode;
}

export default function Card({ title, rightSlot, children }: CardProps) {
  return (
    <section className="rounded-lg border border-border bg-panel p-4 shadow-sm">
      <header className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-gray-900">{title}</h2>
        {rightSlot}
      </header>
      <div>{children}</div>
    </section>
  );
}
