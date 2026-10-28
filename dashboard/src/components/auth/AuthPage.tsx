import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { validateEmail, validatePassword } from '../../utils/security';
import {
  Eye,
  EyeOff,
  Lock,
  Mail,
  User,
  ShieldCheck,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Sparkles,
  Zap,
} from 'lucide-react';

export const AuthPage: React.FC = () => {
  const { login, signup, isLoading } = useAuth();
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');

  // Sign In fields
  const [signInEmail, setSignInEmail] = useState('admin@capermint.com');
  const [signInPassword, setSignInPassword] = useState('Admin@123');
  const [showSignInPassword, setShowSignInPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [signInError, setSignInError] = useState<string | null>(null);

  // Sign Up fields
  const [signUpName, setSignUpName] = useState('');
  const [signUpEmail, setSignUpEmail] = useState('');
  const [signUpPassword, setSignUpPassword] = useState('');
  const [signUpConfirmPassword, setSignUpConfirmPassword] = useState('');
  const [showSignUpPassword, setShowSignUpPassword] = useState(false);
  const [showSignUpConfirmPassword, setShowSignUpConfirmPassword] = useState(false);
  const [signUpError, setSignUpError] = useState<string | null>(null);
  const [signUpSuccess, setSignUpSuccess] = useState<string | null>(null);

  // Forgot Password modal
  const [showForgotModal, setShowForgotModal] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotSubmitted, setForgotSubmitted] = useState(false);

  // Password validation analysis
  const passwordCheck = validatePassword(signUpPassword);

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setSignInError(null);

    if (!signInEmail) {
      setSignInError('Please enter your Capermint admin email.');
      return;
    }
    if (!validateEmail(signInEmail)) {
      setSignInError('Please enter a valid email format (e.g. admin@capermint.com).');
      return;
    }
    if (!signInPassword) {
      setSignInError('Please enter your password.');
      return;
    }

    const res = await login(signInEmail, signInPassword, rememberMe);
    if (!res.success) {
      setSignInError(res.error || 'Authentication failed. Please verify credentials.');
    }
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setSignUpError(null);
    setSignUpSuccess(null);

    if (!signUpName.trim()) {
      setSignUpError('Admin name is required.');
      return;
    }
    if (!validateEmail(signUpEmail)) {
      setSignUpError('Please provide a valid business email address.');
      return;
    }
    if (!passwordCheck.isValid) {
      setSignUpError(passwordCheck.errors[0] || 'Password does not meet required security criteria.');
      return;
    }
    if (signUpPassword !== signUpConfirmPassword) {
      setSignUpError('Passwords do not match. Please re-enter identical passwords.');
      return;
    }

    const res = await signup({
      name: signUpName,
      email: signUpEmail,
      passwordPlain: signUpPassword,
    });

    if (!res.success) {
      setSignUpError(res.error || 'Registration failed.');
    } else {
      setSignUpSuccess('Capermint Administrator registered successfully! Redirecting...');
    }
  };

  const handleQuickDemoFill = () => {
    setSignInEmail('admin@capermint.com');
    setSignInPassword('Admin@123');
    setSignInError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex flex-col justify-center items-center p-4 sm:p-6 lg:p-8 relative overflow-hidden">
      {/* Decorative Brand Accent Gradients */}
      <div className="absolute top-0 -left-20 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 -right-20 w-96 h-96 bg-red-600/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-1/2 left-1/3 w-80 h-80 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Main Container */}
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-2xl border border-slate-100 overflow-hidden grid grid-cols-1 lg:grid-cols-12 relative z-10">
        {/* Left Side: Capermint Brand Showcase */}
        <div className="lg:col-span-5 bg-gradient-to-br from-blue-900 via-slate-900 to-navy-950 p-8 sm:p-10 text-white flex flex-col justify-between relative border-b lg:border-b-0 lg:border-r border-blue-900/50">
          <div>
            {/* Logo Badge */}
            <div className="flex items-center gap-3 mb-8">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-blue-600 via-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
                <span className="font-extrabold text-white text-xl tracking-wider">CP</span>
              </div>
              <div>
                <h1 className="font-bold text-lg leading-tight text-white font-['Outfit']">
                  Capermint Intelligence
                </h1>
                <p className="text-xs text-blue-200 font-medium tracking-wide">Capermint Proprietary Analytics</p>
              </div>
            </div>

            {/* Value Highlights */}
            <div className="space-y-4 my-8">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-sky-400 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-sky-400" />
                Capermint Two-Wheeler Analytics
              </h2>
              <ul className="space-y-3 text-xs text-slate-300">
                <li className="flex items-start gap-2.5">
                  <div className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center shrink-0 mt-0.5">
                    ✓
                  </div>
                  <span>Real-time visitor footfall & AI new vs. returning tracking</span>
                </li>
                <li className="flex items-start gap-2.5">
                  <div className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center shrink-0 mt-0.5">
                    ✓
                  </div>
                  <span>Interactive hourly footfall distribution & peak hour zones</span>
                </li>
                <li className="flex items-start gap-2.5">
                  <div className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center shrink-0 mt-0.5">
                    ✓
                  </div>
                  <span>1-Click visit purpose tagging (Test Ride, Booking, Service)</span>
                </li>
                <li className="flex items-start gap-2.5">
                  <div className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center shrink-0 mt-0.5">
                    ✓
                  </div>
                  <span>Today Average duration with previous week & month comparison</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Quick Demo Fill */}
          <div className="pt-6 border-t border-slate-700/60">
            <p className="text-xs text-slate-400 mb-2 font-medium flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Admin Access:
            </p>
            <button
              type="button"
              onClick={() => {
                setMode('signin');
                handleQuickDemoFill();
              }}
              className="w-full py-2.5 px-3 bg-blue-600/40 hover:bg-blue-600/60 text-white border-2 border-blue-400/80 hover:border-blue-300 rounded-xl text-xs font-semibold shadow-sm transition flex items-center justify-center gap-2 active:scale-[0.99]"
            >
              <Lock className="w-3.5 h-3.5" />
              <span>Fill Admin Demo Credentials</span>
            </button>
          </div>
        </div>

        {/* Right Side: Authentication Forms */}
        <div className="lg:col-span-7 p-8 sm:p-10 flex flex-col justify-center">
          {/* Tabs: Sign In vs Sign Up */}
          <div className="flex border-b border-slate-200 mb-6">
            <button
              type="button"
              onClick={() => {
                setMode('signin');
                setSignInError(null);
              }}
              className={`pb-3 font-semibold text-sm transition-all relative ${
                mode === 'signin' ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'
              }`}
            >
              Admin Sign In
              {mode === 'signin' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 rounded-full" />
              )}
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('signup');
                setSignUpError(null);
                setSignUpSuccess(null);
              }}
              className={`pb-3 font-semibold text-sm ml-8 transition-all relative ${
                mode === 'signup' ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'
              }`}
            >
              Register Admin
              {mode === 'signup' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 rounded-full" />
              )}
            </button>
          </div>

          {/* ==================== ADMIN SIGN IN ==================== */}
          {mode === 'signin' && (
            <form onSubmit={handleSignIn} className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-slate-900 font-['Outfit']">Admin Portal</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Sign in with Capermint administrator credentials to access the intelligence dashboard.
                </p>
              </div>

              {signInError && (
                <div className="p-4 bg-red-50/95 border-2 border-red-300 rounded-2xl text-red-900 text-xs shadow-sm animate-fadeIn space-y-2.5">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle className="w-5 h-5 shrink-0 text-red-600 mt-0.5" />
                    <div>
                      <p className="font-bold text-sm text-red-900">Wrong Password Entered</p>
                      <p className="text-red-800 mt-0.5 leading-relaxed">
                        Please use the correct administrator password:{' '}
                        <span className="font-mono font-bold bg-white text-red-950 px-2 py-0.5 rounded border border-red-300 shadow-2xs">
                          Admin@123
                        </span>
                      </p>
                    </div>
                  </div>
                  <div className="pl-7 flex flex-wrap items-center gap-2 pt-1">
                    <button
                      type="button"
                      onClick={() => {
                        setSignInPassword('Admin@123');
                        setSignInError(null);
                      }}
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold text-xs border border-blue-700 shadow-sm transition flex items-center gap-1.5"
                    >
                      <Lock className="w-3.5 h-3.5 text-blue-200" />
                      <span>Use Correct Password (Admin@123)</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleQuickDemoFill}
                      className="px-3 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded-lg font-semibold text-xs border border-slate-300 shadow-2xs transition"
                    >
                      Auto-fill Both Email & Password
                    </button>
                  </div>
                </div>
              )}

              {/* Email */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Admin Email Address</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <Mail className="w-4 h-4" />
                  </div>
                  <input
                    type="email"
                    value={signInEmail}
                    onChange={(e) => setSignInEmail(e.target.value)}
                    placeholder="admin@capermint.com"
                    className="w-full pl-9 pr-3 py-2.5 text-sm bg-slate-50 border border-slate-300 hover:border-slate-400 rounded-xl focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100 outline-none transition"
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="block text-xs font-semibold text-slate-700">Password</label>
                  <button
                    type="button"
                    onClick={() => setShowForgotModal(true)}
                    className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                  >
                    Forgot password?
                  </button>
                </div>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <Lock className="w-4 h-4" />
                  </div>
                  <input
                    type={showSignInPassword ? 'text' : 'password'}
                    value={signInPassword}
                    onChange={(e) => {
                      setSignInPassword(e.target.value);
                      if (signInError) setSignInError(null);
                    }}
                    placeholder="Enter admin password (e.g. Admin@123)"
                    className={`w-full pl-9 pr-10 py-2.5 text-sm rounded-xl outline-none transition ${
                      signInError
                        ? 'bg-red-50/40 border-2 border-red-400 focus:border-red-500 shadow-2xs'
                        : 'bg-slate-50 border border-slate-300 hover:border-slate-400 focus:bg-white focus:border-blue-600 focus:ring-2 focus:ring-blue-100'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowSignInPassword(!showSignInPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600"
                  >
                    {showSignInPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                {signInError && (
                  <p className="mt-1.5 text-xs text-red-600 font-medium flex items-center gap-1 animate-fadeIn">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0 text-red-500" />
                    <span>Please enter the correct password: <strong className="font-semibold underline">Admin@123</strong></span>
                  </p>
                )}
              </div>

              {/* Remember Me */}
              <div className="flex items-center">
                <input
                  id="remember-me"
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="remember-me" className="ml-2 text-xs text-slate-600 select-none">
                  Remember admin session on this terminal
                </label>
              </div>

              {/* Submit Button with improved border */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold border-2 border-blue-700 shadow-md shadow-blue-500/25 hover:shadow-lg hover:shadow-blue-500/30 active:scale-[0.99] flex items-center justify-center gap-2 transition disabled:opacity-50"
              >
                {isLoading ? (
                  <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    <span>Sign In as Admin</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>

              <p className="text-center text-xs text-slate-500 pt-2">
                Need to create an administrator profile?{' '}
                <button
                  type="button"
                  onClick={() => setMode('signup')}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Register here
                </button>
              </p>
            </form>
          )}

          {/* ==================== REGISTER ADMIN ==================== */}
          {mode === 'signup' && (
            <form onSubmit={handleSignUp} className="space-y-3.5">
              <div>
                <h3 className="text-xl font-bold text-slate-900 font-['Outfit']">Register Capermint Administrator</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Create administrative credentials for the Capermint Showroom Intelligence platform.
                </p>
              </div>

              {signUpError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-xs flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-600" />
                  <span>{signUpError}</span>
                </div>
              )}

              {signUpSuccess && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-xs flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-600" />
                  <span>{signUpSuccess}</span>
                </div>
              )}

              {/* Admin Name */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Full Name</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <User className="w-4 h-4" />
                  </div>
                  <input
                    type="text"
                    value={signUpName}
                    onChange={(e) => setSignUpName(e.target.value)}
                    placeholder="e.g. Vikramaditya Singhania"
                    className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition"
                  />
                </div>
              </div>

              {/* Admin Email */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Capermint Business Email</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <Mail className="w-4 h-4" />
                  </div>
                  <input
                    type="email"
                    value={signUpEmail}
                    onChange={(e) => setSignUpEmail(e.target.value)}
                    placeholder="admin@capermint.com"
                    className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition"
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Admin Password</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <Lock className="w-4 h-4" />
                  </div>
                  <input
                    type={showSignUpPassword ? 'text' : 'password'}
                    value={signUpPassword}
                    onChange={(e) => setSignUpPassword(e.target.value)}
                    placeholder="Min 8 chars, uppercase, lowercase, number, symbol"
                    className="w-full pl-9 pr-10 py-2 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:border-blue-600 outline-none transition"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSignUpPassword(!showSignUpPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600"
                  >
                    {showSignUpPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                {/* Password strength meter */}
                {signUpPassword && (
                  <div className="mt-2 space-y-1.5">
                    <div className="flex gap-1">
                      <div
                        className={`h-1.5 flex-1 rounded-full transition-all ${
                          passwordCheck.score >= 1 ? 'bg-red-500' : 'bg-slate-200'
                        }`}
                      />
                      <div
                        className={`h-1.5 flex-1 rounded-full transition-all ${
                          passwordCheck.score >= 2 ? 'bg-amber-500' : 'bg-slate-200'
                        }`}
                      />
                      <div
                        className={`h-1.5 flex-1 rounded-full transition-all ${
                          passwordCheck.score >= 3 ? 'bg-blue-500' : 'bg-slate-200'
                        }`}
                      />
                      <div
                        className={`h-1.5 flex-1 rounded-full transition-all ${
                          passwordCheck.score >= 4 ? 'bg-emerald-500' : 'bg-slate-200'
                        }`}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-500">
                      <span className={passwordCheck.hasMinLength ? 'text-emerald-600 font-medium' : ''}>
                        {passwordCheck.hasMinLength ? '✓' : '•'} 8+ characters
                      </span>
                      <span
                        className={
                          passwordCheck.hasUppercase && passwordCheck.hasLowercase
                            ? 'text-emerald-600 font-medium'
                            : ''
                        }
                      >
                        {passwordCheck.hasUppercase && passwordCheck.hasLowercase ? '✓' : '•'} Upper & lowercase
                      </span>
                      <span className={passwordCheck.hasNumber ? 'text-emerald-600 font-medium' : ''}>
                        {passwordCheck.hasNumber ? '✓' : '•'} At least 1 number
                      </span>
                      <span className={passwordCheck.hasSpecialChar ? 'text-emerald-600 font-medium' : ''}>
                        {passwordCheck.hasSpecialChar ? '✓' : '•'} Special symbol (!@#$)
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Confirm Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Confirm Password</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <input
                    type={showSignUpConfirmPassword ? 'text' : 'password'}
                    value={signUpConfirmPassword}
                    onChange={(e) => setSignUpConfirmPassword(e.target.value)}
                    placeholder="Re-enter password to confirm"
                    className={`w-full pl-9 pr-10 py-2 text-sm bg-slate-50 border rounded-xl focus:bg-white outline-none transition ${
                      signUpConfirmPassword && signUpPassword !== signUpConfirmPassword
                        ? 'border-red-400 focus:border-red-500'
                        : 'border-slate-200 focus:border-blue-600'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowSignUpConfirmPassword(!showSignUpConfirmPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600"
                  >
                    {showSignUpConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold border-2 border-blue-700 shadow-md shadow-blue-500/25 hover:shadow-lg hover:shadow-blue-500/30 flex items-center justify-center gap-2 transition disabled:opacity-50 mt-2 active:scale-[0.99]"
              >
                {isLoading ? (
                  <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    <span>Register Administrator</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>

              <p className="text-center text-xs text-slate-500 pt-1">
                Already registered?{' '}
                <button
                  type="button"
                  onClick={() => setMode('signin')}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Sign in here
                </button>
              </p>
            </form>
          )}
        </div>
      </div>

      {/* Forgot Password Modal */}
      {showForgotModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200">
            <h4 className="text-lg font-bold text-slate-900 font-['Outfit']">Admin Password Recovery</h4>
            <p className="text-xs text-slate-500 mt-1">
              Enter your Capermint admin email address to receive password reset authorization.
            </p>

            {forgotSubmitted ? (
              <div className="my-5 p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-800">
                <p className="font-semibold mb-1">Reset authorization sent!</p>
                <p>
                  If an admin account exists for <span className="font-mono font-bold">{forgotEmail}</span>, instructions
                  have been dispatched.
                </p>
                <button
                  type="button"
                  onClick={() => {
                    setShowForgotModal(false);
                    setForgotSubmitted(false);
                  }}
                  className="mt-3 w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl text-xs border border-emerald-700 shadow-xs transition"
                >
                  Return to Sign In
                </button>
              </div>
            ) : (
              <div className="my-4 space-y-3">
                <input
                  type="email"
                  value={forgotEmail}
                  onChange={(e) => setForgotEmail(e.target.value)}
                  placeholder="admin@capermint.com"
                  className="w-full px-3 py-2 text-sm border border-slate-300 rounded-xl focus:border-blue-600 outline-none"
                />
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setShowForgotModal(false)}
                    className="flex-1 py-2 border border-slate-300 text-slate-700 font-semibold rounded-xl text-xs hover:bg-slate-50 transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (validateEmail(forgotEmail)) setForgotSubmitted(true);
                    }}
                    className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl text-xs border border-blue-700 shadow-xs transition"
                  >
                    Send Reset Link
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
