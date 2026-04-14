# MemoryPilot Makefile
# Automatise: build > install > sign

BINARY_NAME = MemoryPilot
INSTALL_DIR = $(HOME)/.local/bin

.PHONY: build install sign release clean verify

# Build release
build:
	cargo build --release

# Install + sign in one step
install: build
	cp target/release/$(BINARY_NAME) $(INSTALL_DIR)/$(BINARY_NAME)
	codesign --force --options runtime --sign "Developer ID Application: Soflution ltd (TB8CC687M3)" $(INSTALL_DIR)/$(BINARY_NAME)
	@echo "Installed and signed: $(INSTALL_DIR)/$(BINARY_NAME)"

# Sign only (if already built)
sign:
	codesign --force --options runtime --sign "Developer ID Application: Soflution ltd (TB8CC687M3)" $(INSTALL_DIR)/$(BINARY_NAME)
	@echo "Signed: $(INSTALL_DIR)/$(BINARY_NAME)"

# Build + install + sign (alias)
release: install

# Verify signature
verify:
	codesign --verify --strict $(INSTALL_DIR)/$(BINARY_NAME) && echo "SIGNATURE VALID"
	codesign -dvv $(INSTALL_DIR)/$(BINARY_NAME) 2>&1 | grep -E "Authority|Team"

clean:
	cargo clean
