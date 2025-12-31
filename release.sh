#!/bin/sh
set -e

# Check if we're on the main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Error: Must be on 'main' branch to create a release. Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD -- || ! git diff --staged --quiet; then
    echo "Error: Working directory is not clean. Please commit or stash your changes first."
    exit 1
fi

# Default to patch if no argument provided
BUMP_TYPE=${1:-patch}

# Validate bump type
case "$BUMP_TYPE" in
    major|minor|patch) ;;
    *)
        echo "Error: Invalid bump type '$BUMP_TYPE'. Must be one of: major, minor, patch"
        exit 1
        ;;
esac

# Pull latest changes from remote
echo "Pulling latest changes from remote..."
if ! git pull; then
    echo "Error: Failed to pull latest changes"
    exit 1
fi

# Silent version bumping function with no terminal output
bump_version_silent() {
    local cargo_file=$1
    local current_version=$(grep '^version = ' "$cargo_file" | cut -d'"' -f2)
    if [ -z "$current_version" ]; then
        echo "Error: Could not find version in $cargo_file" >&2
        exit 1
    fi

    # Split version into major, minor, and patch numbers
    local major=$(echo "$current_version" | cut -d. -f1)
    local minor=$(echo "$current_version" | cut -d. -f2)
    local patch=$(echo "$current_version" | cut -d. -f3)

    # Bump version according to type
    case "$BUMP_TYPE" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
    esac

    local new_version="$major.$minor.$patch"

    # Update version in Cargo.toml
    sed -i.bak "s/^version = \"$current_version\"/version = \"$new_version\"/" "$cargo_file"
    rm "${cargo_file}.bak"

    # Print the version without any other message
    echo "$new_version"
}

# First, bump version in the main crate
echo "Updating main crate version..."
MAIN_VERSION=$(bump_version_silent Cargo.toml)
echo "  rstructor version updated to $MAIN_VERSION"

# Set derive crate to the same version as main crate
echo "Syncing derive crate version to $MAIN_VERSION..."
DERIVE_CARGO="rstructor_derive/Cargo.toml"
sed -i.bak "s/^version = \"[0-9.]*\"/version = \"$MAIN_VERSION\"/" "$DERIVE_CARGO"
rm "${DERIVE_CARGO}.bak"
DERIVE_VERSION=$MAIN_VERSION
echo "  rstructor_derive version set to $DERIVE_VERSION"

# Update the dependency reference in main Cargo.toml
echo "Updating dependency reference in main Cargo.toml..."
sed -i.bak "s/rstructor_derive = { version = \"[0-9.]*\"/rstructor_derive = { version = \"$DERIVE_VERSION\"/" Cargo.toml
rm Cargo.toml.bak

# Generate lockfile for the workspace
echo "Updating Cargo.lock..."
cargo generate-lockfile

# Find the previous release tag (looking for the most recent v* tag)
PREVIOUS_TAG=$(git describe --tags --abbrev=0 --match "v*" 2>/dev/null || echo "")
if [ -z "$PREVIOUS_TAG" ]; then
    echo "No previous release tag found. This appears to be the first release."
    CHANGELOG_RANGE="HEAD"
else
    echo "Previous release tag: $PREVIOUS_TAG"
    CHANGELOG_RANGE="$PREVIOUS_TAG..HEAD"
fi

# Generate changelog from git commits
generate_changelog() {
    if [ -z "$PREVIOUS_TAG" ]; then
        # For first release, get all commits
        git log --pretty=format:"- %s (%h)" --no-merges
    else
        # Get commits since last release, excluding version bump commits
        git log --pretty=format:"- %s (%h)" --no-merges "$CHANGELOG_RANGE" | grep -v "Bump version"
    fi
}

CHANGELOG=$(generate_changelog)
if [ -z "$CHANGELOG" ]; then
    CHANGELOG="- No changes to document"
fi

# Create release notes
RELEASE_NOTES=$(cat <<EOF
## Version $MAIN_VERSION

### Changes

$CHANGELOG

### Dependency Versions
- **rstructor**: $MAIN_VERSION
- **rstructor_derive**: $DERIVE_VERSION
EOF
)

# Create git commit and tag for both
git add rstructor_derive/Cargo.toml Cargo.toml
git commit -m "Bump version to $MAIN_VERSION"
git tag -a "v$MAIN_VERSION" -m "Version $MAIN_VERSION"
git tag -a "derive-v$DERIVE_VERSION" -m "Derive Version $DERIVE_VERSION"

echo "Successfully bumped versions:"
echo "  - rstructor_derive: $DERIVE_VERSION"
echo "  - rstructor: $MAIN_VERSION"

# Ask for confirmation before pushing to git
read -p "Would you like to push the changes and tags to git? (y/N) " should_push
if [ "$should_push" = "y" ] || [ "$should_push" = "Y" ]; then
    git push && git push origin "v$MAIN_VERSION" "derive-v$DERIVE_VERSION"
    echo "Successfully pushed changes to git"

    # Create GitHub release with changelog
    echo "Creating GitHub release..."
    echo "$RELEASE_NOTES" | gh release create "v$MAIN_VERSION" \
        --title "v$MAIN_VERSION" \
        --notes-file - \
        --target main
    echo "Successfully created GitHub release"
else
    echo "Skipped pushing to git"
fi

# Ask for confirmation before publishing to crates.io
read -p "Would you like to publish to crates.io? (y/N) " should_publish
if [ "$should_publish" = "y" ] || [ "$should_publish" = "Y" ]; then
    # Verify what will be packaged for derive crate
    echo ""
    echo "Verifying files to be packaged for rstructor_derive..."
    echo "Files that will be included:"
    (cd rstructor_derive && cargo package --list --allow-dirty 2>/dev/null | head -20)
    echo "... (showing first 20 files)"
    read -p "Continue with rstructor_derive publish? (y/N) " continue_derive
    if [ "$continue_derive" != "y" ] && [ "$continue_derive" != "Y" ]; then
        echo "Aborted publishing rstructor_derive"
        exit 1
    fi

    # Publish derive crate first
    echo "Publishing rstructor_derive v$DERIVE_VERSION to crates.io..."
    (cd rstructor_derive && cargo publish)

    # Wait a moment for crates.io to register the new version
    echo "Waiting 15 seconds for crates.io to update..."
    sleep 15

    # Verify what will be packaged for main crate
    echo ""
    echo "Verifying files to be packaged for rstructor..."
    echo "Files that will be included:"
    cargo package --list --allow-dirty 2>/dev/null | head -30
    echo "... (showing first 30 files)"
    read -p "Continue with rstructor publish? (y/N) " continue_main
    if [ "$continue_main" != "y" ] && [ "$continue_main" != "Y" ]; then
        echo "Aborted publishing rstructor"
        exit 1
    fi

    # Then publish main crate
    echo "Publishing rstructor v$MAIN_VERSION to crates.io..."
    cargo publish

    echo "Successfully published both crates to crates.io"
else
    echo "Skipped publishing to crates.io"
fi