'use client';

import PageShell from '@/components/PageShell';
import RequireUser from '@/components/RequireUser';
import VersionPage from '@/components/VersionPage';

export default function VersionRoute() {
  return (
    <RequireUser>
      <PageShell>
        <div className="max-w-7xl mx-auto px-4 py-6">
          <VersionPage />
        </div>
      </PageShell>
    </RequireUser>
  );
}
