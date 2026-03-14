import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "expgraph",
  description: "experiment graph visualization",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet" />
      </head>
      <body style={{
        margin: 0,
        background: "#f0f0f0",
        color: "#333",
        fontFamily: "ui-sans-serif, system-ui, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'",
      }}>
        {children}
      </body>
    </html>
  )
}
