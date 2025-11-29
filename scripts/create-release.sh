#!/bin/bash

# Simple script to create a new release
# Usage: ./scripts/create-release.sh [major|minor|patch]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    print_warning "You're not on the main branch (current: $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    print_warning "You have uncommitted changes:"
    git status -s
    read -p "Commit them before creating release? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        read -p "Enter commit message: " COMMIT_MSG
        git commit -m "$COMMIT_MSG"
        print_success "Changes committed"
    else
        print_warning "Creating release with uncommitted changes"
    fi
fi

# Read current version
if [ -f VERSION ]; then
    CURRENT_VERSION=$(cat VERSION)
else
    CURRENT_VERSION="1.0.0"
    echo "$CURRENT_VERSION" > VERSION
fi

print_step "Current version: $CURRENT_VERSION"

# Parse version
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Determine version bump type
BUMP_TYPE=${1:-patch}  # Default to patch if not specified

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Usage: $0 [major|minor|patch]"
        echo ""
        echo "major: 1.0.0 -> 2.0.0 (breaking changes)"
        echo "minor: 1.0.0 -> 1.1.0 (new features)"
        echo "patch: 1.0.0 -> 1.0.1 (bug fixes)"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"

print_step "New version: $NEW_VERSION"

# Confirm
read -p "Create release v$NEW_VERSION? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Release cancelled"
    exit 1
fi

# Update VERSION file
echo "$NEW_VERSION" > VERSION
print_success "Updated VERSION file"

# Commit version bump
git add VERSION
git commit -m "chore: bump version to v$NEW_VERSION" || print_warning "No changes to commit"

# Create and push tag
print_step "Creating git tag v$NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
print_success "Tag created"

# Push to remote
print_step "Pushing to remote..."
git push origin "$CURRENT_BRANCH"
git push origin "v$NEW_VERSION"
print_success "Pushed to remote"

echo ""
print_success "Release v$NEW_VERSION created successfully! 🎉"
echo ""
echo "📦 GitHub Actions will now:"
echo "   1. Run tests"
echo "   2. Generate changelog"
echo "   3. Create release archives"
echo "   4. Publish GitHub Release"
echo ""
echo "🔗 Check progress at:"
echo "   https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
echo ""
echo "📝 View release when ready:"
echo "   https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/releases/tag/v$NEW_VERSION"
