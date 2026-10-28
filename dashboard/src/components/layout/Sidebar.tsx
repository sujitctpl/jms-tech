import React from 'react';
import {
  LayoutDashboard,
  Users,
  Clock,
  TrendingUp,
  Tags,
  FileSpreadsheet,
  Settings,
  ChevronRight,
  LogOut,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useShowroom } from '../../context/ShowroomContext';

export type ActiveTab =
  | 'overview'
  | 'visitors'
  | 'attendance'
  | 'peak-hours'
  | 'visit-purpose'
  | 'reports'
  | 'settings';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  isMobileOpen: boolean;
  setIsMobileOpen: (open: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  isMobileOpen,
  setIsMobileOpen,
}) => {
  const { user, logout } = useAuth();
  const { metrics } = useShowroom();

  const navigationItems = [
    {
      id: 'overview' as ActiveTab,
      label: 'Overview',
      icon: LayoutDashboard,
      badge: null,
    },
    {
      id: 'visitors' as ActiveTab,
      label: 'Visitors',
      icon: Users,
      badge: metrics.totalVisitors > 0 ? `${metrics.totalVisitors}` : null,
    },
    {
      id: 'attendance' as ActiveTab,
      label: 'Employee Attendance',
      icon: Clock,
      badge: `${metrics.employeesOnDuty}/${metrics.employeesTotal}`,
    },
    {
      id: 'peak-hours' as ActiveTab,
      label: 'Peak Hours',
      icon: TrendingUp,
      badge: 'Peak: 5-6 PM',
      badgeColor: 'bg-amber-100 text-amber-800',
    },
    {
      id: 'visit-purpose' as ActiveTab,
      label: 'Visit Purpose',
      icon: Tags,
      badge: metrics.untaggedCount > 0 ? `${metrics.untaggedCount} untagged` : null,
      badgeColor: 'bg-red-100 text-red-700',
    },
    {
      id: 'reports' as ActiveTab,
      label: 'Reports & Export',
      icon: FileSpreadsheet,
      badge: null,
    },
    {
      id: 'settings' as ActiveTab,
      label: 'Store Operating Hours',
      icon: Settings,
      badge: null,
    },
  ];

  const handleNavClick = (tab: ActiveTab) => {
    setActiveTab(tab);
    setIsMobileOpen(false);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-40 lg:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Permanently Fixed Sidebar */}
      <aside
        className={`fixed top-0 left-0 bottom-0 z-40 w-72 h-screen bg-gradient-to-b from-[#0b1329] via-[#0f172a] to-[#090e1d] border-r border-slate-800/80 flex flex-col justify-between overflow-y-auto transition-transform duration-300 ease-in-out shadow-2xl ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        <div>
          {/* Logo & Header */}
          <div className="p-5 border-b border-slate-800/80">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-600/30 text-white font-extrabold text-sm tracking-wider border border-white/20 shrink-0">
                CP
              </div>
              <div className="overflow-hidden">
                <div className="flex items-center gap-1.5">
                  <h1 className="font-bold text-white text-sm tracking-tight truncate font-['Outfit']">
                    Capermint Intelligence
                  </h1>
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-medium mt-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse shadow-xs shadow-emerald-400/80" />
                  <span>Capermint AI Suite</span>
                </div>
              </div>
            </div>

            {/* Capermint Platform Badge */}
            <div className="mt-4 p-3 bg-slate-800/60 border border-slate-700/60 rounded-xl backdrop-blur-xs">
              <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Showroom Platform</p>
              <p className="text-xs font-bold text-white truncate mt-0.5 font-['Outfit']">
                Capermint Showroom
              </p>
              <div className="mt-1.5 flex items-center justify-between text-[11px]">
                <span className="font-semibold text-blue-400">Capermint Tech</span>
                <span className="text-[10px] bg-blue-500/20 text-blue-300 font-bold px-2 py-0.5 rounded-full border border-blue-400/30">
                  Administrator
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1">
            <p className="px-3 pt-2 pb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">Analytics</p>
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold transition-all group border ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-blue-400/30 shadow-md shadow-blue-600/30'
                      : 'border-transparent text-slate-400 hover:text-white hover:bg-slate-800/60 hover:border-slate-700/50'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon
                      className={`w-4 h-4 transition-transform group-hover:scale-110 ${
                        isActive ? 'text-white' : 'text-slate-400 group-hover:text-blue-400'
                      }`}
                    />
                    <span>{item.label}</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {item.badge && (
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full font-bold transition ${
                          isActive
                            ? 'bg-white/20 text-white border border-white/20'
                            : item.badgeColor
                            ? `${item.badgeColor} border border-amber-300/40`
                            : 'bg-slate-800 text-slate-300 border border-slate-700 group-hover:bg-blue-900/40 group-hover:text-blue-300 group-hover:border-blue-700/60'
                        }`}
                      >
                        {item.badge}
                      </span>
                    )}
                    <ChevronRight
                      className={`w-3.5 h-3.5 transition-transform ${
                        isActive ? 'text-white translate-x-0.5' : 'text-slate-600 opacity-0 group-hover:opacity-100 group-hover:text-slate-400'
                      }`}
                    />
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Section: Admin Profile & Logout */}
        <div className="p-3 border-t border-slate-800/80 bg-slate-900/40">
          <div className="flex items-center justify-between p-2 rounded-xl bg-slate-800/40 border border-slate-700/40">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <img
                src={user?.avatarUrl || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120'}
                alt={user?.name}
                className="w-8 h-8 rounded-full object-cover border border-slate-600 shrink-0"
              />
              <div className="overflow-hidden">
                <p className="text-xs font-bold text-white truncate font-['Outfit']">{user?.name || 'Capermint Admin'}</p>
                <p className="text-[10px] text-slate-400 truncate">{user?.email}</p>
              </div>
            </div>
            <button
              onClick={logout}
              title="Sign Out"
              className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition border border-transparent hover:border-red-500/30 shadow-2xs"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
};
