import React, { useState } from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ShowroomProvider } from './context/ShowroomContext';
import { ErrorBoundary } from './components/common/ErrorBoundary';
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

const VALID_TABS: ActiveTab[] = [
  'overview',
  'visitors',
  'attendance',
  'peak-hours',
  'visit-purpose',
  'reports',
  'settings',
];

const DashboardContent: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const pathTab = location.pathname.replace(/^\//, '').split('/')[0] as ActiveTab;
  const activeTab: ActiveTab = VALID_TABS.includes(pathTab) ? pathTab : 'overview';
  // Keep the query string (e.g. ?range=last7) when switching pages
  const setActiveTab = (tab: ActiveTab) => navigate({ pathname: `/${tab}`, search: location.search });
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
          <Routes>
            <Route path="/" element={<Navigate to={{ pathname: '/overview', search: location.search }} replace />} />
            <Route
              path="/overview"
              element={
                <OverviewView
                  onNavigateTab={(tab) => setActiveTab(tab)}
                  onTagVisitor={(vis) => setSelectedVisitorForTag(vis)}
                  onOpenQuickAdd={() => setIsQuickAddOpen(true)}
                />
              }
            />
            <Route
              path="/visitors"
              element={
                <VisitorsView
                  onTagVisitor={(vis) => setSelectedVisitorForTag(vis)}
                  onOpenQuickAdd={() => setIsQuickAddOpen(true)}
                />
              }
            />
            <Route path="/attendance" element={<AttendanceView />} />
            <Route path="/peak-hours" element={<PeakHoursView />} />
            <Route
              path="/visit-purpose"
              element={<VisitPurposeView onTagVisitor={(vis) => setSelectedVisitorForTag(vis)} />}
            />
            <Route path="/reports" element={<ReportsView />} />
            <Route path="/settings" element={<SettingsView />} />
            <Route path="*" element={<Navigate to="/overview" replace />} />
          </Routes>
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
    <ErrorBoundary>
      <AuthProvider>
        <ShowroomProvider>
          <DashboardContent />
        </ShowroomProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
}

export default App;
