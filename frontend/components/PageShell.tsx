'use client';

import Navbar from '@/components/Navbar';
import SessionDrawer from '@/components/SessionDrawer';

export default function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar />
      <SessionDrawer />
      <main>{children}</main>
    </div>
  );
}
