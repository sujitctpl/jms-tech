import React, { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ShowroomProvider } from './context/ShowroomContext';
import { AuthPage } from './components/auth/AuthPage';
import { Sidebar, ActiveTab } from './components/layout/Sidebar';
import { Navbar } from './components/layout/Navbar';
import { OverviewView } from './components/dashboard/OverviewView';
import { VisitorsView } from './components/visitors/VisitorsView';
import { AttendanceView } from './components/attendance/AttendanceView';
import { PeakHoursView } from './components/peak-hours/PeakHoursView';
import { VisitPurposeView } from './components/visit-purpose/VisitPurposeView';
import { ReportsView } from './components/reports/ReportsView';
import { SettingsView } from './components/settings/SettingsView';
import { TagVisitorModal } from './components/modals/TagVisitorModal';
import { QuickAddVisitorModal } from './components/modals/QuickAddVisitorModal';
import { VisitorRecord } from './types';

const DashboardContent: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  // Modals state
  const [selectedVisitorForTag, setSelectedVisitorForTag] = useState<VisitorRecord | null>(null);
  const [isQuickAddOpen, setIsQuickAddOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center text-white">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-blue-600 via-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30 mb-4 animate-pulse">
          <span className="font-extrabold text-white text-xl">CP</span>
        </div>
        <div className="inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-2" />
        <p className="text-xs text-slate-400 font-medium">Initializing Capermint Showroom Intelligence...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AuthPage />;
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col lg:flex-row text-slate-800">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isMobileOpen={isMobileOpen}
        setIsMobileOpen={setIsMobileOpen}
      />

      {/* Main Content Area (Offset by 72 on desktop for fixed sidebar) */}
      <div className="flex-1 flex flex-col min-w-0 lg:pl-72">
        <Navbar
          onToggleMobileSidebar={() => setIsMobileOpen(true)}
          onOpenQuickAddVisitor={() => setIsQuickAddOpen(true)}
        />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 w-full">
          {activeTab === 'overview' && (
            <OverviewView
              onNavigateTab={(tab) => setActiveTab(tab)}
              onTagVisitor={(vis) => setSelectedVisitorForTag(vis)}
              onOpenQuickAdd={() => setIsQuickAddOpen(true)}
            />
          )}

          {activeTab === 'visitors' && (
            <VisitorsView
              onTagVisitor={(vis) => setSelectedVisitorForTag(vis)}
              onOpenQuickAdd={() => setIsQuickAddOpen(true)}
            />
          )}

          {activeTab === 'attendance' && <AttendanceView />}

          {activeTab === 'peak-hours' && <PeakHoursView />}

          {activeTab === 'visit-purpose' && (
            <VisitPurposeView onTagVisitor={(vis) => setSelectedVisitorForTag(vis)} />
          )}

          {activeTab === 'reports' && <ReportsView />}

          {activeTab === 'settings' && <SettingsView />}
        </main>
      </div>

      {/* Global Modals */}
      <TagVisitorModal
        visitor={selectedVisitorForTag}
        onClose={() => setSelectedVisitorForTag(null)}
      />

      <QuickAddVisitorModal
        isOpen={isQuickAddOpen}
        onClose={() => setIsQuickAddOpen(false)}
      />
    </div>
  );
};

export function App() {
  return (
    <AuthProvider>
      <ShowroomProvider>
        <DashboardContent />
      </ShowroomProvider>
    </AuthProvider>
  );
}

export default App;
