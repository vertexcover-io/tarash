#!/bin/bash

# create_package.sh
# Usage: ./create_package.sh <namespace> <package_name>
# Example: ./create_package.sh tarash gateway

set -e

NAMESPACE=$1
PACKAGE_NAME=$2
PACKAGES_DIR="packages"

# Validate arguments
if [ -z "$NAMESPACE" ] || [ -z "$PACKAGE_NAME" ]; then
    echo "Usage: ./create_package.sh <namespace> <package_name>"
    echo "Example: ./create_package.sh tarash gateway"
    exit 1
fi

# Strip namespace prefix if it exists in package name
# e.g., if PACKAGE_NAME is "tarash-gateway", make it just "gateway"
PACKAGE_NAME="${PACKAGE_NAME#${NAMESPACE}-}"

FULL_PACKAGE_NAME="${NAMESPACE}-${PACKAGE_NAME}"
PACKAGE_PATH="${PACKAGES_DIR}/${PACKAGE_NAME}"
ROOT_PYPROJECT="pyproject.toml"

# Convert hyphen to underscore for Python module name
PACKAGE_NAME_UNDERSCORE="${PACKAGE_NAME//-/_}"

echo "Creating package: ${FULL_PACKAGE_NAME}"

# Check if package already exists
if [ -d "$PACKAGE_PATH" ]; then
    echo "Error: Package directory already exists at ${PACKAGE_PATH}"
    exit 1
fi

# Create package with uv init --lib
uv init "${PACKAGE_PATH}" --lib

# Create namespace package structure
mkdir -p "${PACKAGE_PATH}/src/${NAMESPACE}/${PACKAGE_NAME_UNDERSCORE}"

# Move generated files to namespace structure
mv "${PACKAGE_PATH}/src/${PACKAGE_NAME_UNDERSCORE}/"* "${PACKAGE_PATH}/src/${NAMESPACE}/${PACKAGE_NAME_UNDERSCORE}/"
rmdir "${PACKAGE_PATH}/src/${PACKAGE_NAME_UNDERSCORE}"

# Update package pyproject.toml - change name
sed -i.bak "s/name = \"${PACKAGE_NAME}\"/name = \"${FULL_PACKAGE_NAME}\"/" "${PACKAGE_PATH}/pyproject.toml"

# Update package pyproject.toml - add hatch wheel config
if ! grep -q "\[tool.hatch.build.targets.wheel\]" "${PACKAGE_PATH}/pyproject.toml"; then
    cat >> "${PACKAGE_PATH}/pyproject.toml" << EOF

[tool.hatch.build.targets.wheel]
packages = ["src/${NAMESPACE}"]
EOF
fi

rm -f "${PACKAGE_PATH}/pyproject.toml.bak"

# Add to root [tool.uv.sources]
if ! grep -q "${FULL_PACKAGE_NAME}" "$ROOT_PYPROJECT"; then
    sed -i.bak "/\[tool.uv.sources\]/a\\
${FULL_PACKAGE_NAME} = { workspace = true }
" "$ROOT_PYPROJECT"
    rm -f "${ROOT_PYPROJECT}.bak"
fi

echo ""
echo "✓ Package ${FULL_PACKAGE_NAME} created"
echo "  from ${NAMESPACE}.${PACKAGE_NAME_UNDERSCORE} import ..."
echo ""
echo "Run 'uv sync' to update workspace."
