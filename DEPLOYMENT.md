# Heroku Deployment Guide

This guide explains how to deploy the ML service to Heroku.

## Prerequisites

1. Heroku account
2. Heroku CLI installed
3. Git repository with ml_model directory

## Environment Variables

Set the following environment variables in Heroku for database connection:

```bash
heroku config:set DB_HOST=your_database_host
heroku config:set DB_USER=your_database_user
heroku config:set DB_PASSWORD=your_database_password
heroku config:set DB_NAME=your_database_name
```

### Example (for Hostinger database):
```bash
heroku config:set DB_HOST=srv1322.hstgr.io
heroku config:set DB_USER=u520834156_userPmoc
heroku config:set DB_PASSWORD=NzkN5arIO7@
heroku config:set DB_NAME=u520834156_DBpmoc25
```

## Deployment Steps

### 1. Navigate to ml_model directory
```bash
cd ml_model
```

### 2. Initialize Heroku app (if not already done)
```bash
heroku create your-app-name
```

Or link to existing app:
```bash
heroku git:remote -a your-app-name
```

### 3. Set environment variables
```bash
heroku config:set DB_HOST=your_host
heroku config:set DB_USER=your_user
heroku config:set DB_PASSWORD=your_password
heroku config:set DB_NAME=your_database
```

### 4. Deploy to Heroku
```bash
git add .
git commit -m "Deploy ML service to Heroku"
git push heroku master
```

Or if using main branch:
```bash
git push heroku main
```

### 5. Check logs
```bash
heroku logs --tail
```

## Files Required for Deployment

- `service.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku process file (uses gunicorn)
- `runtime.txt` - Python version specification
- Model files (`.pkl` files) - ML models

## Service Endpoints

Once deployed, your service will be available at:
- `https://your-app-name.herokuapp.com/`

### Health Check
- `GET /health` - Check if service is running

### Main Endpoints
- `POST /analyze` - Analyze couple responses
- `POST /train` - Train ML models
- `GET /training-status` - Check training status

## Troubleshooting

### Database Connection Issues
- Verify environment variables are set correctly
- Check database host allows connections from Heroku IPs
- Ensure database credentials are correct

### Model Loading Issues
- Ensure all `.pkl` files are committed to git
- Check that model files are in the correct directory

### Build Failures
- Check `requirements.txt` for correct package versions
- Verify Python version in `runtime.txt` matches Heroku supported versions

## Notes

- The service uses gunicorn for production (configured in Procfile)
- Database connections use environment variables for security
- If database connection fails, the service falls back to hardcoded categories
- Model files should be committed to git (they're binary but necessary for the service)

