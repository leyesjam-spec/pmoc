# Quick Heroku Deployment Instructions

Your endpoint: `https://endpoint-pmoc-a0a6708d039f.herokuapp.com/`

## Step 1: Install Heroku CLI (if not installed)

Download from: https://devcenter.heroku.com/articles/heroku-cli

## Step 2: Login to Heroku

```bash
heroku login
```

## Step 3: Navigate to ml_model directory

```bash
cd ml_model
```

## Step 4: Link to your Heroku app

Based on your endpoint, your app name is likely `endpoint-pmoc`. Link it:

```bash
heroku git:remote -a endpoint-pmoc
```

If that doesn't work, try:
```bash
heroku git:remote -a endpoint-pmoc-a0a6708d039f
```

Or check your existing remotes:
```bash
git remote -v
```

## Step 5: Set Environment Variables

Set your database configuration (replace with your actual database credentials):

```bash
heroku config:set DB_HOST=srv1322.hstgr.io
heroku config:set DB_USER=u520834156_userPmoc
heroku config:set DB_PASSWORD=NzkN5arIO7@
heroku config:set DB_NAME=u520834156_DBpmoc25
```

**Important:** Use your actual database credentials. The service will fall back to hardcoded categories if the database connection fails.

## Step 6: Deploy to Heroku

```bash
git push heroku master
```

Or if your default branch is `main`:
```bash
git push heroku main
```

## Step 7: Check Deployment Status

```bash
heroku logs --tail
```

## Step 8: Verify Service is Running

Visit: `https://endpoint-pmoc-a0a6708d039f.herokuapp.com/health`

Or test with curl:
```bash
curl https://endpoint-pmoc-a0a6708d039f.herokuapp.com/health
```

## Troubleshooting

### If deployment fails:
1. Check logs: `heroku logs --tail`
2. Verify all files are committed: `git status`
3. Ensure model files (.pkl) are in the repository
4. Check that requirements.txt has all dependencies

### If service doesn't start:
1. Check Procfile is correct: `cat Procfile` (should show `web: gunicorn service:app`)
2. Verify gunicorn is in requirements.txt
3. Check runtime.txt has correct Python version

### Database Connection Issues:
- The service will work even if database connection fails (uses fallback categories)
- Verify environment variables: `heroku config`
- Check database allows connections from Heroku IPs

## What Changed for Heroku Deployment:

1. ✅ **Database Configuration**: Now uses environment variables (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
2. ✅ **Production Server**: Changed from Flask dev server to gunicorn
3. ✅ **Procfile**: Updated to use gunicorn
4. ✅ **Requirements**: Added gunicorn to requirements.txt
5. ✅ **Port Configuration**: Already configured to use Heroku's PORT environment variable

## Next Steps After Deployment:

1. Update your PHP application to point to the Heroku endpoint
2. Test the `/analyze` endpoint with sample data
3. Monitor logs for any errors: `heroku logs --tail`

