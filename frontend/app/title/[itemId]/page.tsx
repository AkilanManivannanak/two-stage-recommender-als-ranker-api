'use client';

import PageShell from '@/components/PageShell';
import RequireUser from '@/components/RequireUser';
import TitlePage from '@/components/TitlePage';

export default function TitleRoute() {
  return (
    <RequireUser>
      <PageShell>
        <TitlePage />
      </PageShell>
    </RequireUser>
  );
}
