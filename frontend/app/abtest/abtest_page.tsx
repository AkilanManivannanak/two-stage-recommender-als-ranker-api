'use client'
import dynamic from 'next/dynamic'
const ABDashboard = dynamic(() => import('@/components/ABDashboard'), { ssr: false })
export default function ABTestRoute() { return <ABDashboard /> }
