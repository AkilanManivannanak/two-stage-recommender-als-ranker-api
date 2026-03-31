'use client';

import PageShell from '@/components/PageShell';
import RequireUser from '@/components/RequireUser';
import MetricsPage from '@/components/MetricsPage';

export default function MetricsRoute() {
  return (
    <RequireUser>
      <PageShell>
        <div className="max-w-7xl mx-auto px-4 py-6">
          <MetricsPage />
        </div>
      </PageShell>
    </RequireUser>
  );
}
