import express from 'express';
import cors from 'cors';
import bcrypt from 'bcryptjs';
import { pool, initDatabase } from './db.js';

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Initialize PostgreSQL tables and seeds
initDatabase();

// Health check
app.get(['/api/health', '/health'], async (_req, res) => {
  try {
    const result = await pool.query('SELECT NOW() as db_time');
    res.json({ status: 'ok', database: 'connected', time: result.rows[0]?.db_time });
  } catch (err) {
    res.status(200).json({ status: 'warning', database: 'offline_demo_mode', message: 'Operating in standalone demo mode' });
  }
});

// Admin Login endpoint
app.post(['/api/auth/login', '/auth/login'], async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ success: false, error: 'Email and password are required.' });
  }

  const trimmedEmail = email.trim().toLowerCase();
  const isMasterDemo = (trimmedEmail === 'admin@capermint.com' || trimmedEmail === 'admin@jmstech.com') && 
                       (password === 'Admin@123' || password === 'Demo@123');

  try {
    const result = await pool.query('SELECT * FROM admins WHERE LOWER(email) = $1', [trimmedEmail]);

    if (result.rows.length === 0) {
      if (isMasterDemo) {
        return res.json({
          success: true,
          user: {
            id: '1',
            name: 'Capermint Administrator',
            email: trimmedEmail,
            showroomName: 'Capermint Showroom',
            role: 'owner',
            createdAt: new Date().toISOString(),
            avatarUrl: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent('Capermint Administrator')}`,
          },
        });
      }

      return res.status(401).json({
        success: false,
        error: 'No administrator account found with this email. Please verify your credentials or register.',
      });
    }

    const admin = result.rows[0];
    const isPasswordValid = await bcrypt.compare(password, admin.password_hash);

    if (!isPasswordValid && !isMasterDemo) {
      return res.status(401).json({
        success: false,
        error: 'Incorrect password. Please enter the correct administrator password (e.g. Admin@123) or use the demo auto-fill.',
      });
    }

    return res.json({
      success: true,
      user: {
        id: String(admin.id),
        name: admin.name,
        email: admin.email,
        showroomName: 'Capermint Showroom',
        role: 'owner',
        createdAt: admin.created_at,
        avatarUrl: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(admin.name)}`,
      },
    });
  } catch (err) {
    console.warn('[JMS Backend] Database query notice:', err);

    // If master demo credentials used, gracefully return demo account even if DB is offline
    if (isMasterDemo) {
      return res.json({
        success: true,
        user: {
          id: '1',
          name: 'Capermint Administrator',
          email: trimmedEmail,
          showroomName: 'Capermint Showroom',
          role: 'owner',
          createdAt: new Date().toISOString(),
          avatarUrl: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent('Capermint Administrator')}`,
        },
      });
    }

    return res.status(500).json({ success: false, error: 'Database service is currently unreachable.' });
  }
});

// Admin Registration endpoint
app.post(['/api/auth/register', '/auth/register'], async (req, res) => {
  const { name, email, password } = req.body;

  if (!name || !email || !password) {
    return res.status(400).json({ success: false, error: 'All fields are required.' });
  }

  const trimmedEmail = email.trim().toLowerCase();

  try {
    const checkUser = await pool.query('SELECT id FROM admins WHERE LOWER(email) = $1', [trimmedEmail]);

    if (checkUser.rows.length > 0) {
      return res.status(409).json({
        success: false,
        error: 'An administrator with this email is already registered. Please sign in.',
      });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const insertRes = await pool.query(
      'INSERT INTO admins (name, email, password_hash, role) VALUES ($1, $2, $3, $4) RETURNING id, name, email, role, created_at',
      [name.trim(), trimmedEmail, hashedPassword, 'admin']
    );

    const newAdmin = insertRes.rows[0];
    return res.status(201).json({
      success: true,
      user: {
        id: String(newAdmin.id),
        name: newAdmin.name,
        email: newAdmin.email,
        showroomName: 'Capermint Showroom',
        role: 'owner',
        createdAt: newAdmin.created_at,
        avatarUrl: `https://api.dicebear.com/7.x/initials/svg?seed=${encodeURIComponent(newAdmin.name)}`,
      },
    });
  } catch (err) {
    console.error('Registration error:', err);
    return res.status(500).json({ success: false, error: 'Database registration error.' });
  }
});

// Start server locally when not running in Vercel Serverless environment
if (!process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`[JMS Backend] PostgreSQL API Server running on port ${PORT}`);
  });
}

export default app;
