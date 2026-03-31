'use client'

import dynamic from 'next/dynamic'

const AIStackPage = dynamic(() => import('@/components/AIStackPage'), { ssr: false })

export default function AIStackRoute() {
  return <AIStackPage />
}
