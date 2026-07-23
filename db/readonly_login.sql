-- ============================================================================
-- readonly_login.sql
-- ============================================================================
-- Provisions a least-privilege reporting login for AgniAI's text-to-SQL
-- backend (sql_executor.py). This login is the PRIMARY safety control for
-- that backend — sql_executor.validate_sql() is defense-in-depth, not a
-- substitute. Never point SQL_READONLY_CONN at anything more privileged.
--
-- Run once, by a DBA, against the SQL Server instance hosting
-- DB_Agni. Idempotent: safe to re-run.
--
-- After running, put the resulting connection string in SQL_READONLY_CONN
-- (see .env.example), e.g.:
--   SQL_READONLY_CONN=Driver={ODBC Driver 17 for SQL Server};Server=<host>;Database=DB_Agni;UID=agniai_reporting;PWD=<password>;Encrypt=yes;TrustServerCertificate=no;
-- ============================================================================

USE master;
GO

-- ── 1. Server login ─────────────────────────────────────────────────────────
-- CHANGE_ME: replace the password below before running, and rotate it via
-- your normal secrets process — do not commit the real password anywhere.
IF NOT EXISTS (
    SELECT 1
    FROM sys.server_principals
    WHERE
        name = 'agniai_reporting'
) BEGIN CREATE LOGIN agniai_reporting
WITH
    PASSWORD = N 'CHANGE_ME_STRONG_PASSWORD!',
    CHECK_POLICY = ON,
    CHECK_EXPIRATION = ON;

END

USE DB_Agni;
GO

-- ── 2. Database user mapped to the login ────────────────────────────────────
IF NOT EXISTS (
    SELECT 1
    FROM sys.database_principals
    WHERE
        name = 'agniai_reporting'
) BEGIN CREATE USER agniai_reporting FOR LOGIN agniai_reporting;

END

-- ── 3. Grant read-only role ──────────────────────────────────────────────────
ALTER ROLE db_datareader ADD MEMBER agniai_reporting;
GO

-- ── 4. Hard denies — sensitive columns/tables, even though db_datareader
--       would otherwise allow SELECT on them. These mirror
--       sql_executor.DENIED_COLUMNS / DENIED_TABLES and A.4 in the schema
--       reference; they must never be removed.
DENY SELECT ON dbo.UserMaster (Password) TO agniai_reporting;
GO
DENY SELECT ON dbo.LoginToken TO agniai_reporting;
GO
DENY SELECT ON dbo.DefaultLog TO agniai_reporting;
GO

-- ── 5. Explicit denies on write/DDL/ownership roles ─────────────────────────
-- db_datareader does not grant these by default, but deny them explicitly so
-- a future role-membership change elsewhere cannot silently escalate this
-- login's privileges without also touching this script.
IF EXISTS (
    SELECT 1
    FROM sys.database_principals
    WHERE
        name = 'db_datawriter'
) BEGIN ALTER ROLE db_datawriter
DROP MEMBER agniai_reporting;

END

DENY INSERT, UPDATE, DELETE, EXECUTE ON DATABASE::DB_Agni TO agniai_reporting;
GO
DENY ALTER, CONTROL, CREATE TABLE, CREATE PROCEDURE, CREATE VIEW ON DATABASE::DB_Agni TO agniai_reporting;
GO

-- ── 6. Verification queries (run manually after provisioning) ──────────────
-- Should succeed:
--   EXECUTE AS USER = 'agniai_reporting'; SELECT TOP 1 * FROM AgniveerMaster; REVERT;
-- Should fail with a permission error:
--   EXECUTE AS USER = 'agniai_reporting'; SELECT Password FROM UserMaster; REVERT;
--   EXECUTE AS USER = 'agniai_reporting'; SELECT * FROM LoginToken; REVERT;
--   EXECUTE AS USER = 'agniai_reporting'; DELETE FROM AgniveerMaster; REVERT;