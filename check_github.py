#!/usr/bin/env python3
"""
Check if the GitHub repository is correctly set up.
"""

import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def check_git():
    """Check if git is installed and initialized."""
    if not run_command("which git"):
        print("Git is not installed. Please install git first.")
        return False
    
    if not os.path.exists(".git"):
        print("Git repository is not initialized. Initializing now...")
        run_command("git init")
        print("Git repository initialized.")
    
    return True

def check_git_config():
    """Check if git user and email are configured."""
    user_name = run_command("git config --get user.name")
    user_email = run_command("git config --get user.email")
    
    if not user_name or not user_email:
        print("\nGit user or email not configured. Please configure with:")
        print('  git config --global user.name "Your Name"')
        print('  git config --global user.email "your.email@example.com"')
        return False
    
    print(f"\nGit configured for user: {user_name} <{user_email}>")
    return True

def check_github_remote():
    """Check if a GitHub remote is configured."""
    remotes = run_command("git remote -v")
    
    if not remotes:
        print("\nNo git remotes found. You need to add a GitHub remote.")
        setup_remote = input("Would you like to set up a GitHub remote now? (y/n): ")
        
        if setup_remote.lower() == 'y':
            github_url = input("\nEnter your GitHub repository URL: ")
            if github_url:
                run_command(f"git remote add github {github_url}")
                print(f"Added GitHub remote: {github_url}")
                return True
            else:
                print("No URL provided.")
                return False
        return False
    
    if "github" not in remotes:
        print("\nNo 'github' remote found. Current remotes:")
        print(remotes)
        
        setup_remote = input("Would you like to add a GitHub remote? (y/n): ")
        if setup_remote.lower() == 'y':
            github_url = input("\nEnter your GitHub repository URL: ")
            if github_url:
                run_command(f"git remote add github {github_url}")
                print(f"Added GitHub remote: {github_url}")
                return True
            else:
                print("No URL provided.")
                return False
    else:
        github_url = run_command("git remote get-url github")
        print(f"\nGitHub remote configured: {github_url}")
        return True

def main():
    """Main function to check GitHub repository setup."""
    print("Checking GitHub repository setup...\n")
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py") or not os.path.isdir("ml4s"):
        print("Error: This script must be run from the ML4S-Benchmark directory.")
        sys.exit(1)
    
    # Check git
    if not check_git():
        sys.exit(1)
    
    # Check git config
    if not check_git_config():
        print("Please configure git and run this script again.")
        sys.exit(1)
    
    # Check GitHub remote
    if not check_github_remote():
        print("\nPlease set up a GitHub remote before pushing.")
        print("You can do this with: git remote add github <your-github-url>")
        sys.exit(1)
    
    print("\nEverything looks good! You can run the push_to_github.sh script to push your changes.")
    print("./push_to_github.sh")

if __name__ == "__main__":
    main() 