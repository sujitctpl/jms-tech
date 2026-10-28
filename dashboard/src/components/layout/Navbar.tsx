import React, { useState } from 'react';
import {
  Menu,
  Bell,
  Calendar,
  Clock,
  PlusCircle,
  ChevronDown,
  LogOut,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useShowroom } from '../../context/ShowroomContext';
import { DateFilterOption } from '../../types';
import { APP_ENV, IS_LOCAL } from '../../config/env';

interface NavbarProps {
  onToggleMobileSidebar: () => void;
  onOpenQuickAddVisitor: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleMobileSidebar, onOpenQuickAddVisitor }) => {
  const { user, logout } = useAuth();
  const {
    storeHours,
    isStoreCurrentlyOpen,
    dateFilter,
    setDateFilter,
    metrics,
    lastSimulatedEvent,
  } = useShowroom();

  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showNotificationMenu, setShowNotificationMenu] = useState(false);

  const dateFilterLabels: Record<DateFilterOption, string> = {
    today: 'Today',
    yesterday: 'Yesterday',
    last7: 'Last 7 Days',
    last30: 'Last 30 Days',
    custom: 'Custom Range',
  };

  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-xl border-b border-slate-200/80 px-4 sm:px-6 lg:px-8 py-3.5 shadow-xs">
      <div className="flex items-center justify-between gap-3 sm:gap-4 w-full">
        {/* Left Section: Mobile Toggle & Store Status */}
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleMobileSidebar}
            className="lg:hidden p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-xl transition border border-slate-200 hover:border-slate-300 shadow-2xs shrink-0"
            aria-label="Open navigation"
          >
            <Menu className="w-5 h-5" />
          </button>

          {/* Store Hours & Dynamic Open/Closed Status */}
          <div className="flex items-center gap-2.5">
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border transition shadow-2xs ${
                isStoreCurrentlyOpen
                  ? 'bg-emerald-50/90 text-emerald-800 border-emerald-300/80'
                  : 'bg-rose-50/90 text-rose-800 border-rose-300/80'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  isStoreCurrentlyOpen ? 'bg-emerald-500 animate-pulse shadow-xs shadow-emerald-500/80' : 'bg-rose-500'
                }`}
              />
              <span className="font-['Outfit'] tracking-tight font-bold">
                {isStoreCurrentlyOpen ? 'Showroom OPEN' : 'Showroom CLOSED'}
              </span>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 text-xs text-slate-500 font-medium pl-1">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>
                Operating Hours: <span className="font-bold text-slate-700">{storeHours.openingTime} - {storeHours.closingTime}</span>
              </span>
            </div>

            {/* Single Variable Environment Badge */}
            <div
              className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-extrabold uppercase tracking-wider border shadow-2xs ${
                IS_LOCAL
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                  : 'bg-amber-50 text-amber-700 border-amber-300'
              }`}
              title={`Switch between LOCAL and PROD via VITE_APP_ENV in dashboard/.env`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${IS_LOCAL ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
              <span>ENV: {APP_ENV}</span>
            </div>
          </div>
        </div>

        {/* Right Section: Date Filter, Quick Entry, Notifications, Profile */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Date Range Selector */}
          <div className="relative hidden md:block">
            <div className="flex items-center bg-slate-100/90 rounded-xl p-1 border border-slate-200 shadow-2xs">
              <Calendar className="w-3.5 h-3.5 text-slate-500 ml-2 mr-1 shrink-0" />
              {(['today', 'yesterday', 'last7', 'last30'] as DateFilterOption[]).map((opt) => (
                <button
                  key={opt}
                  onClick={() => setDateFilter(opt)}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-lg border transition-all ${
                    dateFilter === opt
                      ? 'bg-white text-blue-700 border-slate-300/80 shadow-xs'
                      : 'border-transparent text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {dateFilterLabels[opt]}
                </button>
              ))}
            </div>
          </div>

          {/* Quick Record Walk-in / Entry */}
          <button
            onClick={onOpenQuickAddVisitor}
            className="flex items-center gap-1.5 px-3.5 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-xl text-xs font-semibold border border-blue-500/30 shadow-sm shadow-blue-600/25 transition active:scale-[0.98] shrink-0"
          >
            <PlusCircle className="w-4 h-4" />
            <span className="hidden sm:inline">Log Walk-in</span>
          </button>

          {/* Notifications Bell */}
          <div className="relative">
            <button
              onClick={() => setShowNotificationMenu(!showNotificationMenu)}
              className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-xl transition relative border border-slate-200 hover:border-slate-300 shadow-2xs"
              title="Notifications"
            >
              <Bell className="w-4 h-4" />
              {metrics.untaggedCount > 0 && (
                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 rounded-full ring-2 ring-white animate-pulse" />
              )}
            </button>

            {/* Notification Popover */}
            {showNotificationMenu && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-2xl shadow-xl border border-slate-200/90 p-4 z-50">
                <div className="flex items-center justify-between pb-2 border-b border-slate-100 mb-2">
                  <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Live System Alerts</h4>
                  <span className="text-[10px] bg-blue-100 text-blue-800 font-bold px-1.5 py-0.5 rounded">
                    {metrics.untaggedCount} Action Needed
                  </span>
                </div>

                <div className="space-y-2.5 text-xs">
                  {metrics.untaggedCount > 0 && (
                    <div className="p-2.5 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                      <div>
                        <p className="font-semibold text-amber-900">Untagged Visitors on Floor</p>
                        <p className="text-[11px] text-amber-700 mt-0.5">
                          {metrics.untaggedCount} visitors awaiting purpose assignment.
                        </p>
                      </div>
                    </div>
                  )}

                  {lastSimulatedEvent && (
                    <div className="p-2.5 bg-blue-50 border border-blue-200 rounded-xl flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
                      <div>
                        <p className="font-semibold text-blue-900">Footfall Event</p>
                        <p className="text-[11px] text-blue-700 mt-0.5">{lastSimulatedEvent}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* User Profile & Logout */}
          <div className="relative">
            <button
              onClick={() => setShowProfileMenu(!showProfileMenu)}
              className="flex items-center gap-2 p-1.5 rounded-xl hover:bg-slate-100 transition border border-slate-200 hover:border-slate-300 shadow-2xs"
            >
              <img
                src={user?.avatarUrl || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120'}
                alt={user?.name}
                className="w-7 h-7 rounded-full object-cover border border-slate-300"
              />
              <div className="hidden md:block text-left">
                <span className="text-xs font-bold text-slate-800 block leading-tight">{user?.name}</span>
                <span className="text-[10px] text-slate-500 font-semibold">Administrator</span>
              </div>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            </button>

            {/* Profile Dropdown */}
            {showProfileMenu && (
              <div className="absolute right-0 mt-2 w-56 bg-white rounded-2xl shadow-xl border border-slate-200/90 p-3 z-50 animate-fadeIn">
                <div className="p-2 border-b border-slate-100 mb-2">
                  <p className="text-xs font-bold text-slate-900">{user?.name}</p>
                  <p className="text-[11px] text-slate-500 truncate">{user?.email}</p>
                  <span className="inline-block text-[10px] bg-blue-100 text-blue-800 font-bold px-1.5 py-0.5 rounded mt-1">
                    Capermint Administrator
                  </span>
                </div>

                <button
                  onClick={logout}
                  className="w-full flex items-center gap-2 p-2 text-xs font-semibold text-red-600 hover:bg-red-50 rounded-xl transition border border-transparent hover:border-red-200"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Sign Out</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
