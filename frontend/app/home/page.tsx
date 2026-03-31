'use client';

import PageShell from '@/components/PageShell';
import RequireUser from '@/components/RequireUser';
import HomeScreen from '@/components/HomeScreen';

export default function HomePage() {
  return (
    <RequireUser>
      <PageShell>
        <HomeScreen />
      </PageShell>
    </RequireUser>
  );
}
