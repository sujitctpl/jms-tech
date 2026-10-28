/**
 * Environment Configuration for JMS / Capermint Dashboard
 * Controlled by VITE_APP_ENV in .env:
 * - 'LOCAL': Uses local FastAPI (http://localhost:8000) and local Auth (http://localhost:5000)
 * - 'PROD': Uses production API endpoint (placeholder/configurable)
 */

export type AppEnvironment = 'LOCAL' | 'PROD';

const rawEnv = (import.meta.env.VITE_APP_ENV || 'LOCAL').trim().toUpperCase();

export const APP_ENV: AppEnvironment = rawEnv === 'PROD' ? 'PROD' : 'LOCAL';
export const IS_LOCAL = APP_ENV === 'LOCAL';
export const IS_PROD = APP_ENV === 'PROD';

// FastAPI Base URL (Camera workers, Footfall analytics, Health)
export const FASTAPI_BASE_URL = IS_LOCAL
  ? (import.meta.env.VITE_LOCAL_API_URL || 'http://localhost:8000')
  : (import.meta.env.VITE_PROD_API_URL || '');

// Express Auth Base URL
export const AUTH_BASE_URL = IS_LOCAL
  ? (import.meta.env.VITE_LOCAL_AUTH_URL || 'http://localhost:5000')
  : (import.meta.env.VITE_PROD_API_URL || '');

export const ENV_CONFIG = {
  env: APP_ENV,
  isLocal: IS_LOCAL,
  isProd: IS_PROD,
  fastApiUrl: FASTAPI_BASE_URL,
  authUrl: AUTH_BASE_URL,
  prodUrl: import.meta.env.VITE_PROD_API_URL || '',
};
