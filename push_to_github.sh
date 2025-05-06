#!/bin/bash

# Script to prepare and push ML4S-Benchmark to GitHub
# Make sure you run this script from inside the ML4S-Benchmark directory

# Exit on error
set -e

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "ml4s" ]; then
    echo "Error: Please run this script from inside the ML4S-Benchmark directory"
    exit 1
fi

# Check if git is configured
if [ -z "$(git config --get user.name)" ] || [ -z "$(git config --get user.email)" ]; then
    echo "Git user name or email not configured. Please configure with:"
    echo "git config --global user.name \"Your Name\""
    echo "git config --global user.email \"your.email@example.com\""
    exit 1
fi

# Check for GitHub remote
REMOTE_EXISTS=$(git remote -v | grep -c github || true)
if [ $REMOTE_EXISTS -eq 0 ]; then
    echo "No GitHub remote found. Please add your GitHub repository URL:"
    read -p "Enter GitHub URL (e.g., https://github.com/yourusername/repository.git): " GITHUB_URL
    
    if [ -z "$GITHUB_URL" ]; then
        echo "No URL provided. Exiting."
        exit 1
    fi
    
    git remote add github "$GITHUB_URL"
    echo "Added GitHub remote."
fi

# Make sure we have the latest changes
git status

# Ask user for commit message
echo ""
echo "Please enter a commit message for your changes:"
read -p "Commit message: " COMMIT_MESSAGE

if [ -z "$COMMIT_MESSAGE" ]; then
    COMMIT_MESSAGE="Update ML4S-Benchmark code"
fi

# Stage required files
echo "Adding files to git..."

# Core Python package
git add ml4s/*.py
git add ml4s/tasks/**/*.py
git add ml4s/abstasks/*.py

# Build and setup files
git add setup.py
git add README.md
git add .gitignore

# Add example results.json files but exclude large response caches
find results -name "results.json" -exec git add {} \;

# Ask about including results files
echo ""
read -p "Include all result files? (y/n): " INCLUDE_RESULTS

if [[ $INCLUDE_RESULTS =~ ^[Yy]$ ]]; then
    # Include all result JSON files but not response caches
    echo "Including all result files..."
    find results -name "*.json" -exec git add {} \;
fi

# Commit changes
echo "Committing changes with message: $COMMIT_MESSAGE"
git commit -m "$COMMIT_MESSAGE"

# Push to GitHub
echo "Pushing to GitHub..."
git push github main

echo ""
echo "Changes pushed to GitHub successfully!"
echo "GitHub URL: $(git remote get-url github)" 