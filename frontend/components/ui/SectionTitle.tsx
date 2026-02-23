interface SectionTitleProps {
  title: string;
}

export default function SectionTitle({ title }: SectionTitleProps) {
  return <h1 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-700">{title}</h1>;
}
