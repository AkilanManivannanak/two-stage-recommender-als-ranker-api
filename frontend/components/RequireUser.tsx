'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAppState } from '@/lib/store';

export default function RequireUser({ children }: { children: React.ReactNode }) {
  const { activeUser } = useAppState();
  const router = useRouter();

  useEffect(() => {
    if (!activeUser) router.push('/');
  }, [activeUser, router]);

  if (!activeUser) return null;
  return <>{children}</>;
}
