# 📦 Release Guide for Heritage Document Recommender

This guide explains how to create releases for this project in the simplest way possible.

## 🎯 Quick Start

**To create a new release, just run:**

```bash
./scripts/create-release.sh patch
```

That's it! GitHub Actions will handle the rest automatically.

---

## 📝 Release Types

Choose the right release type based on your changes:

### 🐛 **Patch Release** (Bug fixes)
- Example: `1.0.0` → `1.0.1`
- Use when: Fixing bugs, small improvements, no new features
- Command:
  ```bash
  ./scripts/create-release.sh patch
  ```

### ✨ **Minor Release** (New features)
- Example: `1.0.0` → `1.1.0`
- Use when: Adding new features, backward compatible changes
- Command:
  ```bash
  ./scripts/create-release.sh minor
  ```

### 💥 **Major Release** (Breaking changes)
- Example: `1.0.0` → `2.0.0`
- Use when: Major redesign, breaking API changes, removing features
- Command:
  ```bash
  ./scripts/create-release.sh major
  ```

---

## 🔄 What Happens Automatically

When you run the release script:

1. ✅ **Version bumped** in VERSION file
2. ✅ **Git tag created** (e.g., `v2.0.0`)
3. ✅ **Pushed to GitHub**
4. ✅ **GitHub Actions triggered** which:
   - Runs all tests
   - Generates changelog from commits
   - Creates `.tar.gz` and `.zip` archives
   - Publishes GitHub Release with download links

---

## 🚀 Step-by-Step Example

Let's say you fixed some bugs and want to release version 2.0.1:

```bash
# 1. Make sure all your changes are committed
git add .
git commit -m "fix: resolve search query bug"

# 2. Run the release script
./scripts/create-release.sh patch

# 3. Confirm when prompted
# Current version: 2.0.0
# New version: 2.0.1
# Create release v2.0.1? (y/N) y

# 4. Watch the magic happen! ✨
```

**After a few minutes, check:**
- GitHub Actions: https://github.com/YOUR_USERNAME/heritage_doc_recomm/actions
- New Release: https://github.com/YOUR_USERNAME/heritage_doc_recomm/releases

---

## 📋 Manual Release (Alternative)

If you prefer doing it manually without the script:

```bash
# 1. Update VERSION file
echo "2.0.1" > VERSION

# 2. Commit the version bump
git add VERSION
git commit -m "chore: bump version to v2.0.1"

# 3. Create and push tag
git tag -a v2.0.1 -m "Release v2.0.1"
git push origin main
git push origin v2.0.1
```

GitHub Actions will still automatically create the release!

---

## 🔍 Understanding Version Numbers

Version format: `MAJOR.MINOR.PATCH`

```
2.0.1
│ │ │
│ │ └─── Patch (bug fixes)
│ └───── Minor (new features)
└─────── Major (breaking changes)
```

**Examples:**
- Fixed a typo → `2.0.1` → `2.0.2` (patch)
- Added dark mode → `2.0.2` → `2.1.0` (minor)
- Rewrote entire system → `2.1.0` → `3.0.0` (major)

---

## 📦 What's Included in Releases

Each release includes:
- **Source code** (automatic GitHub archives)
- **Custom archives** (.tar.gz and .zip) without:
  - Git files
  - Virtual environments
  - Python cache files
  - Large data files
- **Changelog** (auto-generated from git commits)
- **Release notes**

---

## ✍️ Writing Good Commit Messages

Since changelog is auto-generated from commits, write clear messages:

### Good Examples ✅
```bash
git commit -m "feat: add learning to rank ensemble"
git commit -m "fix: resolve memory leak in SimRank"
git commit -m "docs: update installation guide"
git commit -m "perf: optimize FAISS indexing"
```

### Bad Examples ❌
```bash
git commit -m "stuff"
git commit -m "fix"
git commit -m "asdf"
git commit -m "WIP"
```

**Commit message prefixes:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `perf:` - Performance improvements
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

---

## 🛠️ Troubleshooting

### Problem: Script says "You're not on main branch"
**Solution:** Switch to main first:
```bash
git checkout main
git pull
./scripts/create-release.sh patch
```

### Problem: "You have uncommitted changes"
**Solution:** Commit or stash them:
```bash
git add .
git commit -m "your message"
# OR
git stash
```

### Problem: Tag already exists
**Solution:** Delete the tag locally and remotely:
```bash
git tag -d v2.0.1
git push origin :refs/tags/v2.0.1
```

### Problem: GitHub Actions failing
**Solution:** Check the Actions tab on GitHub for error details:
- URL: `https://github.com/YOUR_USERNAME/heritage_doc_recomm/actions`
- Look for red ❌ marks
- Click to see logs

---

## 📊 Viewing Releases

**All releases:** https://github.com/YOUR_USERNAME/heritage_doc_recomm/releases

**Latest release:** https://github.com/YOUR_USERNAME/heritage_doc_recomm/releases/latest

**Download specific version:**
```bash
wget https://github.com/YOUR_USERNAME/heritage_doc_recomm/archive/refs/tags/v2.0.0.tar.gz
```

---

## 🎓 Best Practices

1. **Test before releasing** - Make sure everything works
2. **One release per feature** - Don't bundle too many changes
3. **Write clear commit messages** - They become your changelog
4. **Release regularly** - Don't wait months between releases
5. **Keep VERSION file updated** - Let the script handle it
6. **Check GitHub Actions** - Make sure automation succeeds

---

## 📞 Need Help?

- Check GitHub Actions logs for errors
- Review this guide
- Look at previous successful releases
- Contact: akchhya1108@gmail.com

---

## 🎉 That's It!

Creating releases is now as simple as:

```bash
./scripts/create-release.sh patch
```

Happy releasing! 🚀
