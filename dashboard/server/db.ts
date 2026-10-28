import 'dotenv/config';
import { Pool } from 'pg';
import bcrypt from 'bcryptjs';

const connectionString = process.env.DATABASE_URL || process.env.POSTGRES_URL;

export const pool = new Pool(
  connectionString
    ? {
        connectionString,
        ssl: { rejectUnauthorized: false },
      }
    : {
        user: process.env.PGUSER || 'postgres',
        host: process.env.PGHOST || 'localhost',
        database: process.env.PGDATABASE || 'jms_showroom_db',
        password: process.env.PGPASSWORD || 'postgres',
        port: parseInt(process.env.PGPORT || '5432', 10),
      }
);

export async function initDatabase() {
  try {
    const client = await pool.connect();
    console.log('[PostgreSQL] Connected to jms_showroom_db successfully.');

    // Create admins table
    await client.query(`
      CREATE TABLE IF NOT EXISTS admins (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        role VARCHAR(50) DEFAULT 'admin',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );
    `);

    // Check if default admin exists
    const checkCapermint = await client.query('SELECT * FROM admins WHERE email = $1', ['admin@capermint.com']);
    if (checkCapermint.rows.length === 0) {
      const defaultHash = await bcrypt.hash('Admin@123', 10);
      await client.query(
        'INSERT INTO admins (name, email, password_hash, role) VALUES ($1, $2, $3, $4)',
        ['Capermint Administrator', 'admin@capermint.com', defaultHash, 'admin']
      );
      console.log('[PostgreSQL] Default administrator account seeded: admin@capermint.com / Admin@123');
    }

    const checkRes = await client.query('SELECT * FROM admins WHERE email = $1', ['admin@jmstech.com']);
    if (checkRes.rows.length === 0) {
      const defaultHash = await bcrypt.hash('Admin@123', 10);
      await client.query(
        'INSERT INTO admins (name, email, password_hash, role) VALUES ($1, $2, $3, $4)',
        ['Capermint Administrator', 'admin@jmstech.com', defaultHash, 'admin']
      );
    }

    client.release();
  } catch (error) {
    console.error('[PostgreSQL] Database initialization error:', error);
  }
}
