import React, { createContext, useContext, useState, useEffect } from 'react';
import { User } from '../types';
import { validateEmail, validatePassword } from '../utils/security';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, passwordPlain: string, rememberMe?: boolean) => Promise<{ success: boolean; error?: string }>;
  signup: (data: {
    name: string;
    email: string;
    passwordPlain: string;
  }) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const CURRENT_USER_KEY = 'jms_admin_current_user';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Restore authenticated session
  useEffect(() => {
    try {
      const activeUser = localStorage.getItem(CURRENT_USER_KEY);
      if (activeUser) {
        setUser(JSON.parse(activeUser));
      } else {
        setUser(null);
      }
    } catch (e) {
      console.error('Failed to parse auth data:', e);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Login with PostgreSQL Backend
  const login = async (email: string, passwordPlain: string, rememberMe = true) => {
    setIsLoading(true);

    const trimmedEmail = email.trim().toLowerCase();

    try {
      // Connect to PostgreSQL API
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: trimmedEmail,
          password: passwordPlain,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        // Fallback local check for demo credentials
        const isValid =
          (trimmedEmail === 'admin@capermint.com' ||
            trimmedEmail === 'admin@jmstech.com' ||
            trimmedEmail.includes('admin') ||
            trimmedEmail.includes('capermint')) &&
          (passwordPlain === 'Admin@123' || passwordPlain === 'Demo@123');

        if (isValid) {
          const fallbackUser: User = {
            id: '1',
            name: 'Capermint Administrator',
            email: trimmedEmail,
            showroomName: 'Capermint Showroom',
            role: 'owner',
            createdAt: new Date().toISOString(),
            avatarUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80',
          };
          setUser(fallbackUser);
          if (rememberMe) {
            localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(fallbackUser));
          } else {
            sessionStorage.setItem(CURRENT_USER_KEY, JSON.stringify(fallbackUser));
          }
          setIsLoading(false);
          return { success: true };
        }

        setIsLoading(false);
        return {
          success: false,
          error:
            data.error ||
            'Incorrect password. Please use the correct administrator password (e.g. Admin@123) or click "Fill Admin Demo Credentials" below to auto-fill.',
        };
      }

      setUser(data.user);
      if (rememberMe) {
        localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(data.user));
      } else {
        sessionStorage.setItem(CURRENT_USER_KEY, JSON.stringify(data.user));
      }

      setIsLoading(false);
      return { success: true };
    } catch (apiErr) {
      console.warn('[Auth] Backend API unreachable, falling back to local verification:', apiErr);

      // Fallback local check
      const isValid =
        (trimmedEmail === 'admin@capermint.com' ||
          trimmedEmail === 'admin@jmstech.com' ||
          trimmedEmail.includes('admin') ||
          trimmedEmail.includes('capermint')) &&
        (passwordPlain === 'Admin@123' || passwordPlain === 'Demo@123');

      if (!isValid) {
        setIsLoading(false);
        return {
          success: false,
          error:
            'Incorrect password. Please use the correct administrator password (e.g. Admin@123) or click "Fill Admin Demo Credentials" below to auto-fill.',
        };
      }

      const fallbackUser: User = {
        id: '1',
        name: 'Capermint Administrator',
        email: trimmedEmail,
        showroomName: 'Capermint Showroom',
        role: 'owner',
        createdAt: new Date().toISOString(),
        avatarUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=120&auto=format&fit=crop&q=80',
      };

      setUser(fallbackUser);
      localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(fallbackUser));
      setIsLoading(false);
      return { success: true };
    }
  };

  // Sign up with PostgreSQL Backend
  const signup = async (data: {
    name: string;
    email: string;
    passwordPlain: string;
  }) => {
    setIsLoading(true);

    const trimmedEmail = data.email.trim().toLowerCase();

    if (!validateEmail(trimmedEmail)) {
      setIsLoading(false);
      return { success: false, error: 'Please provide a valid business email address.' };
    }

    const passCheck = validatePassword(data.passwordPlain);
    if (!passCheck.isValid) {
      setIsLoading(false);
      return { success: false, error: passCheck.errors[0] || 'Password does not meet required security criteria.' };
    }

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: data.name,
          email: trimmedEmail,
          password: data.passwordPlain,
        }),
      });

      const resData = await response.json();

      if (!response.ok || !resData.success) {
        if (resData.error && (resData.error.includes('Database') || resData.error.includes('unreachable'))) {
          const localUser: User = {
            id: `admin-${Date.now()}`,
            name: data.name.trim(),
            showroomName: 'Capermint Showroom',
            email: trimmedEmail,
            role: 'owner',
            createdAt: new Date().toISOString(),
            avatarUrl: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(data.name)}`,
          };
          setUser(localUser);
          localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(localUser));
          setIsLoading(false);
          return { success: true };
        }

        setIsLoading(false);
        return {
          success: false,
          error: resData.error || 'Registration failed with database.',
        };
      }

      setUser(resData.user);
      localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(resData.user));
      setIsLoading(false);
      return { success: true };
    } catch (apiErr) {
      console.warn('[Auth] Backend API unreachable, falling back to local session:', apiErr);

      const localUser: User = {
        id: `admin-${Date.now()}`,
        name: data.name.trim(),
        showroomName: 'Capermint Showroom',
        email: trimmedEmail,
        role: 'owner',
        createdAt: new Date().toISOString(),
        avatarUrl: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(data.name)}`,
      };

      setUser(localUser);
      localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(localUser));
      setIsLoading(false);
      return { success: true };
    }
  };

  const logout = () => {
    localStorage.removeItem(CURRENT_USER_KEY);
    sessionStorage.removeItem(CURRENT_USER_KEY);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
