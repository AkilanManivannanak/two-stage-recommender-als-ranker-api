import type { Metadata } from 'next'
import './globals.css'
import { AppProvider } from '@/lib/store'

export const metadata: Metadata = {
  title: 'CineWave — AI-Powered Recommendations',
  description: 'Real-time personalized streaming recommendations powered by ML',
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='4' fill='%23080c14'/><text y='24' x='4' font-size='22' font-family='Georgia'>🌊</text></svg>"
  }
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="bg-cine-bg text-cine-text font-body antialiased min-h-screen">
        <AppProvider>
          {children}
        </AppProvider>
      </body>
    </html>
  )
}
