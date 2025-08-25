#!/bin/bash
# HydrAI-SWE Systemd Service Installation Script

set -e

echo "🚀 Installing HydrAI-SWE Data Pipeline Automation..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root"
   exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    echo "❌ systemctl not found. This script requires systemd."
    exit 1
fi

# Create systemd user directory if it doesn't exist
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

echo "📂 Systemd user directory: $SYSTEMD_USER_DIR"

# Copy service files
echo "📋 Installing service files..."

# Copy service file
cp "$SCRIPT_DIR/hydrai-pipeline.service" "$SYSTEMD_USER_DIR/"
echo "  ✅ hydrai-pipeline.service"

# Copy timer file
cp "$SCRIPT_DIR/hydrai-pipeline.timer" "$SYSTEMD_USER_DIR/"
echo "  ✅ hydrai-pipeline.timer"

# Update service file with correct paths
sed -i "s|/home/sean/hydrai_swe|$PROJECT_ROOT|g" "$SYSTEMD_USER_DIR/hydrai-pipeline.service"
sed -i "s|sean|$USER|g" "$SYSTEMD_USER_DIR/hydrai-pipeline.service"

echo "🔧 Updated service file paths for user: $USER"

# Make script executable
chmod +x "$PROJECT_ROOT/scripts/auto_pipeline_update.py"
echo "🔒 Made update script executable"

# Reload systemd user daemon
echo "🔄 Reloading systemd user daemon..."
systemctl --user daemon-reload

# Enable and start timer
echo "🚀 Enabling and starting timer..."
systemctl --user enable hydrai-pipeline.timer
systemctl --user start hydrai-pipeline.timer

# Check status
echo "📊 Checking service status..."
systemctl --user status hydrai-pipeline.timer --no-pager

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Service Information:"
echo "  Service: hydrai-pipeline.service"
echo "  Timer:  hydrai-pipeline.timer"
echo "  Status: $(systemctl --user is-active hydrai-pipeline.timer)"
echo "  Next:   $(systemctl --user list-timers --no-pager | grep hydrai-pipeline || echo 'No timer info')"
echo ""
echo "🔧 Management Commands:"
echo "  Check status:    systemctl --user status hydrai-pipeline.timer"
echo "  View logs:       journalctl --user -u hydrai-pipeline.service"
echo "  Manual run:      systemctl --user start hydrai-pipeline.service"
echo "  Stop timer:      systemctl --user stop hydrai-pipeline.timer"
echo "  Disable timer:   systemctl --user disable hydrai-pipeline.timer"
echo ""
echo "📁 Log files:"
echo "  Systemd logs:    journalctl --user -u hydrai-pipeline.service"
echo "  Application:     $PROJECT_ROOT/logs/pipeline_updates.log"
echo ""
echo "🌐 The timer will automatically update data sources every 6 hours."
echo "   You can modify the schedule in: $SYSTEMD_USER_DIR/hydrai-pipeline.timer"
