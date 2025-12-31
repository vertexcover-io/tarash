# Claude Code Guidelines

This document contains guidelines for Claude Code when working on this project.

## Git Commit Guidelines

### Commit Message Format

- **Use GitHub emoji** at the start of commit messages to categorize changes:
  - ✨ `:sparkles:` New features
  - 🐛 `:bug:` Bug fixes
  - 📝 `:memo:` Documentation updates
  - ♻️ `:recycle:` Refactoring
  - ⚡ `:zap:` Performance improvements
  - ✅ `:white_check_mark:` Tests
  - 🔧 `:wrench:` Configuration changes
  - 🎨 `:art:` Code style/formatting
  - 🔥 `:fire:` Code removal
  - 🚀 `:rocket:` Deployment/release
  - 🔒 `:lock:` Security fixes

### Commit Message Style

- **Be concise**: Focus on the primary change in the commit
- **No AI attribution**: Do not mention "Claude Code" or AI assistance in commit messages
- **Primary change only**: Describe the main feature/bug/change, not every file modification
- **Imperative mood**: Use "Add feature" not "Added feature"

### Examples

Good:
```
✨ Add workspace configuration for monorepo
🐛 Fix dependency resolution in package loader
📝 Update installation instructions
♻️ Restructure error handling
```

Bad:
```
✨ Add workspace configuration, update pyproject.toml, modify README.md, and add CLAUDE.md

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

Updated multiple files including workspace setup
```

### Commit Workflow

**CRITICAL: Always confirm before committing**

When creating any commit, you MUST follow this process:

1. **Show the user what will be committed**:
   - List all files that will be included in the commit
   - Display the proposed commit message
   - Show relevant diffs if helpful for context
   - **Do NOT include untracked files** unless user explicitly requests them

2. **Wait for explicit user confirmation**:
   - Do NOT execute `git commit` until the user approves
   - User must explicitly confirm (e.g., "yes", "ok", "commit it")

3. **Only then create the commit**:
   - Use the approved message exactly as shown
   - Keep commits atomic - one logical change per commit

4. **Focus on impact**: What does this change accomplish, not how it was done
